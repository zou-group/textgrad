from textgrad import logger
from textgrad.engine import EngineLM
from typing import List, Set, Dict
import httpx
from collections import defaultdict
from functools import partial
from .config import SingletonBackwardEngine
from .utils.image_utils import is_valid_url
from typing import Union

class Variable:
    def __init__(
        self,
        value: Union[str, bytes] = "",
        image_path: str = "",
        predecessors: List['Variable']=None,
        requires_grad: bool=True,
        *,
        role_description: str):
        """The main thing. Nodes in the computation graph. Really the heart and soul of textgrad.

        :param value: The string value of this variable, defaults to "". In the future, we'll go multimodal, for sure!
        :type value: str or bytes, optional
        :param image_path: The path to the image file, defaults to "". If present we will read from disk or download the image.
        :type image_path: str, optional
        :param predecessors: predecessors of this variable in the computation graph, defaults to None. Here, for instance, if we have a prompt -> response through an LLM call, we'd call the prompt the predecessor, and the response the successor. 
        :type predecessors: List[Variable], optional
        :param requires_grad: Whether this variable requires a gradient, defaults to True. If False, we'll not compute the gradients on this variable.
        :type requires_grad: bool, optional
        :param role_description: The role of this variable. We find that this has a huge impact on the optimization performance, and being specific often helps quite a bit!
        :type role_description: str
        """

        if predecessors is None:
            predecessors = []
        
        _predecessor_requires_grad = [v for v in predecessors if v.requires_grad]
        
        if (not requires_grad) and (len(_predecessor_requires_grad) > 0):
            raise Exception("If the variable does not require grad, none of its predecessors should require grad."
                            f"In this case, following predecessors require grad: {_predecessor_requires_grad}")
        
        assert type(value) in [str, bytes, int], "Value must be a string, int, or image (bytes). Got: {}".format(type(value))
        if isinstance(value, int):
            value = str(value)
        # We'll currently let "empty variables" slide, but we'll need to handle this better in the future.
        # if value == "" and image_path == "":
        #    raise ValueError("Please provide a value or an image path for the variable")
        if value != "" and image_path != "":
            raise ValueError("Please provide either a value or an image path for the variable, not both.")

        if image_path != "":
            if is_valid_url(image_path):
                self.value = httpx.get(image_path).content
            else:
                with open(image_path, 'rb') as file:
                    self.value = file.read()
        else:
            self.value = value
            
        self.gradients: Set[Variable] = set()
        self.gradients_context: Dict[Variable, str] = defaultdict(lambda: None)
        self.grad_fn = None
        self.role_description = role_description
        self.predecessors = set(predecessors)
        self.requires_grad = requires_grad
        self._reduce_meta = []
        
        if requires_grad and (type(value) == bytes):
            raise ValueError("Gradients are not yet supported for image inputs. Please provide a string input instead.")

    def __repr__(self):
        return f"Variable(value={self.value}, role={self.get_role_description()}, grads={self.gradients})"

    def __str__(self):
        return str(self.value)

    def __add__(self, to_add):
        # For now, let's just assume variables can be passed to models
        if isinstance(to_add, Variable):
            ### Somehow handle the addition of variables
            total = Variable(
                value=self.value + to_add.value,
                # Add the predecessors of both variables
                predecessors=[self, to_add],
                # Communicate both of the roles
                role_description=f"{self.role_description} and {to_add.role_description}",
                # We should require grad if either of the variables require grad
                requires_grad=(self.requires_grad | to_add.requires_grad),
            )
            total.set_grad_fn(partial(
                _backward_idempotent,
                variables=total.predecessors,
                summation=total,
            ))
            return total
        else:
            return to_add.__add__(self)

    def set_role_description(self, role_description):
        self.role_description = role_description

    def reset_gradients(self):
        self.gradients = set()
        self.gradients_context = dict()
        self._reduce_meta = []

    def get_role_description(self) -> str:
        return self.role_description

    def get_short_value(self, n_words_offset: int=10) -> str:
        """
        Returns a short version of the value of the variable. We sometimes use it during optimization, when we want to see the value of the variable, but don't want to see the entire value.
        This is sometimes to save tokens, sometimes to reduce repeating very long variables, such as code or solutions to hard problems.
        :param n_words_offset: The number of words to show from the beginning and the end of the value.
        :type n_words_offset: int
        """
        words = self.value.split(" ")
        if len(words) <= 2 * n_words_offset:
            return self.value
        short_value = " ".join(words[:n_words_offset]) + " (...) " + " ".join(words[-n_words_offset:])
        return short_value

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value

    def set_grad_fn(self, grad_fn):
        self.grad_fn = grad_fn

    def get_grad_fn(self):
        return self.grad_fn

    def get_gradient_text(self) -> str:
        """Aggregates and returns the gradients on a variable."""
        
        return "\n".join([g.value for g in self.gradients])
    
    def backward(self, engine: EngineLM = None):
        """
        Backpropagate gradients through the computation graph starting from this variable.

        :param engine: The backward engine to use for gradient computation. If not provided, the global engine will be used.
        :type engine: EngineLM, optional

        :raises Exception: If no backward engine is provided and no global engine is set.
        :raises Exception: If both an engine is provided and the global engine is set.
        """
        if ((engine is None) and (SingletonBackwardEngine().get_engine() is None)):
            raise Exception("No backward engine provided. Either provide an engine as the argument to this call, or use `textgrad.set_backward_engine(engine)` to set the backward engine.")
        elif ((engine is not None) and (SingletonBackwardEngine().get_engine() is not None)):
            raise Exception("Both an engine is provided and the global engine is set. Be careful when doing this.")
        
        backward_engine = engine if engine else SingletonBackwardEngine().get_engine()
        """Taken from https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py"""
        # topological order all the predecessors in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for predecessor in v.predecessors:
                    build_topo(predecessor)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        # TODO: we should somehow ensure that we do not have cases such as the predecessors of a variable requiring a gradient, but the variable itself not requiring a gradient

        self.gradients = set()
        for v in reversed(topo):
            if v.requires_grad:
                v.gradients = _check_and_reduce_gradients(v, backward_engine)
                if v.get_grad_fn() is not None:
                    v.grad_fn(backward_engine=backward_engine)
                    
    def generate_graph(self, print_gradients: bool=False):
        """
        Generates a computation graph starting from the variable itself.

        :param print_gradients: A boolean indicating whether to print gradients in the graph.
        :return: A visualization of the computation graph.
        """
        try:
            from graphviz import Digraph
        except ImportError:
            raise ImportError("Please install graphviz to visualize the computation graphs. You can install it using `pip install graphviz`.")
        
        def wrap_text(text, width=40):
            """Wraps text at a given number of characters using HTML line breaks."""
            words = text.split()
            wrapped_text = ""
            line = ""
            for word in words:
                if len(line) + len(word) + 1 > width:
                    wrapped_text += line + "<br/>"
                    line = word
                else:
                    if line:
                        line += " "
                    line += word
            wrapped_text += line
            return wrapped_text
        
        def wrap_and_escape(text, width=40):
            return wrap_text(text.replace("<", "&lt;").replace(">", "&gt;"), width)
        
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for predecessor in v.predecessors:
                    build_topo(predecessor)
                topo.append(v)

        def get_grad_fn_name(name):
            ws = name.split(" ")
            ws = [w for w in ws if "backward" in w]
            return " ".join(ws)
            
        build_topo(self)

        graph = Digraph(comment='Computation Graph starting from {}'.format(self.role_description))
        graph.attr(rankdir='TB')  # Set the graph direction from top to bottom
        graph.attr(ranksep='0.2')  # Adjust the spacing between ranks
        graph.attr(bgcolor='lightgrey')  # Set the background color of the graph
        graph.attr(fontsize='7.5')  # Set the font size of the graph
        
        for v in reversed(topo):
            # Add each node to the graph
            label_color = 'darkblue'
            
            node_label = (
                f"<b><font color='{label_color}'>Role: </font></b> {wrap_and_escape(v.role_description.capitalize())}"
                f"<br/><b><font color='{label_color}'>Value: </font></b> {wrap_and_escape(v.value)}"
            )
            

            if v.grad_fn is not None:
                node_label += f"<br/><b><font color='{label_color}'>Grad Fn: </font></b> {wrap_and_escape(get_grad_fn_name(str(v.grad_fn)))}"

            if v._reduce_meta != []:
                node_label += f"<br/><b><font color='{label_color}'>Reduce Meta: </font></b> {wrap_and_escape(str(v._reduce_meta))}"

            if print_gradients:
                node_label += f"<br/><b><font color='{label_color}'>Gradients: </font></b> {wrap_and_escape(v.get_gradient_text())}"
            # Update the graph node with modern font and better color scheme
            graph.node(
                str(id(v)), 
                label=f"<{node_label}>", 
                shape='rectangle', 
                style='filled', 
                fillcolor='lavender', 
                fontsize='8', 
                fontname="Arial", 
                margin='0.1', 
                pad='0.1', 
                width='1.2',
            )
            # Add forward edges from predecessors to the parent
            for predecessor in v.predecessors:
                graph.edge(str(id(predecessor)), str(id(v)))
        
        return graph


def _check_and_reduce_gradients(variable: Variable, backward_engine: EngineLM) -> Set[Variable]:
    """
    Check and reduce gradients for a given variable.

    This function checks if the gradients of the variable need to be reduced based on the reduction groups
    specified in the variable's metadata. If reduction is required, it performs the reduction operation
    for each reduction group and returns the reduced gradients.
    
    For example, we do things like averaging the gradients using this functionality.

    :param variable: The variable for which gradients need to be checked and reduced.
    :type variable: Variable
    :param backward_engine: The backward engine used for gradient computation.
    :type backward_engine: EngineLM

    :return: The reduced gradients for the variable.
    :rtype: Set[Variable]
    """
    if variable._reduce_meta == []:
        return variable.gradients
    if variable.get_gradient_text() == "":
        return variable.gradients
    
    if len(variable.gradients) == 1:
        return variable.gradients
    
    id_to_gradient_set = defaultdict(set)
    id_to_op = {} # Note: there must be a way to ensure that the op is the same for all the variables with the same id
    
    # Go through each gradient, group them by their reduction groups
    for gradient in variable.gradients:
        for reduce_item in gradient._reduce_meta:
            id_to_gradient_set[reduce_item["id"]].add(gradient)
            id_to_op[reduce_item["id"]] = reduce_item["op"]
    # For each reduction group, perform the reduction operation
    new_gradients = set()
    for group_id, gradients in id_to_gradient_set.items():
        logger.info(f"Reducing gradients for group {group_id}", extra={"gradients": gradients})
        new_gradients.add(id_to_op[group_id](gradients, backward_engine))
    
    return new_gradients


def _backward_idempotent(variables: List[Variable], summation: Variable, backward_engine: EngineLM):
    """
    Perform an idempotent backward pass e.g. for textgrad.sum or Variable.__add__.
    In particular, this function backpropagates the gradients of the `summation` variable to all the variables in the `variables` list.

    :param variables: The list of variables to backpropagate the gradients to.
    :type variables: List[Variable]
    :param summation: The variable representing the summation operation.
    :type summation: Variable
    :param backward_engine: The backward engine used for backpropagation.
    :type backward_engine: EngineLM

    :return: None

    :notes:
        - The idempotent backward pass is used for textgrad.sum or Variable.__add__ operations.
        - The gradients of the `summation` variable are backpropagated to all the variables in the `variables` list.
        - The feedback from each variable is stored in their respective gradients.
        - The feedback from the `summation` variable is combined and stored in the `summation_gradients` variable.
        - The feedback from each variable is later used for feedback propagation to other variables in the computation graph.
        - The `_reduce_meta` attribute of the `summation` variable is used to reduce the feedback if specified.
    """
    summation_gradients = summation.get_gradient_text()
    for variable in variables: 
        if summation_gradients == "":
            variable_gradient_value = ""
        else:
            variable_gradient_value = f"Here is the combined feedback we got for this specific {variable.get_role_description()} and other variables: {summation_gradients}."
            
        logger.info(f"Idempotent backward", extra={"v_gradient_value": variable_gradient_value, 
                                                   "summation_role": summation.get_role_description()})

        var_gradients = Variable(value=variable_gradient_value, 
                                 role_description=f"feedback to {variable.get_role_description()}")
        variable.gradients.add(var_gradients)
        
        if summation._reduce_meta != []:
            var_gradients._reduce_meta.extend(summation._reduce_meta)
            variable._reduce_meta.extend(summation._reduce_meta)

        variable.gradients.add(Variable(value=variable_gradient_value, 
                                        role_description=f"feedback to {variable.get_role_description()}"))
        