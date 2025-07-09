# 
prompt = """
Read the following question, generate the background knowledge as the context information that could be helpful for answering the question.

Question: Which property do these three objects have in common? 

Options: (A) hard (B) soft (C) yellow

Metadata: {'pid': 43, 'has_image': True, 'grade': 4, 'subject': 'natural science', 'topic': 'physics', 'category': 'Materials', 'skill': 'Compare properties of objects'} 

Detected text in the image: ['handkerchief', 'slippers', 'leisure suit']

Knowledge:
- This question is about comparing the properties of three objects: a handkerchief, slippers, and a leisure suit.
- The objects are related to the topic of physics and the skill of comparing properties of objects.
- Properties of objects can include physical characteristics such as color, texture, shape, size, weight, and material.

Question: The diagrams below show two pure samples of gas in identical closed, rigid containers. Each colored ball represents one gas particle. Both samples have the same number of particles. 

Options: (A) neither; the samples have the same temperature (B) sample A (C) sample B

Metadata: {'pid': 19, 'has_image': True, 'grade': 8, 'subject': 'natural science', 'topic': 'physics', 'category': 'Particle motion and energy', 'skill': 'Identify how particle motion affects temperature and pressure'}

Knowledge:
- The temperature of a substance depends on the average kinetic energy of the particles in the substance. 
- The higher the average kinetic energy of the particles, the higher the temperature of the substance.
- The kinetic energy of a particle is determined by its mass and speed. 
- For a pure substance, the greater the mass of each particle in the substance and the higher the average speed of the particles, the higher their average kinetic energy. 

Question: Think about the magnetic force between the magnets in each pair. Which of the following statements is true?

Context: The images below show two pairs of magnets. The magnets in different pairs do not affect each other. All the magnets shown are made of the same material, but some of them are different shapes.  

Options: (A) The magnitude of the magnetic force is greater in Pair 1. (B) The magnitude of the magnetic force is greater in Pair 2. (C) The magnitude of the magnetic force is the same in both pairs.

Metadata: {'pid': 270, 'has_image': True, 'grade': 6, 'subject': 'natural science', 'topic': 'physics', 'category': 'Velocity, acceleration, and forces', 'skill': 'Compare magnitudes of magnetic forces'}

Knowledge:
- Magnets can pull or push on each other without touching. When magnets attract, they pull together. When magnets repel, they push apart. These pulls and pushes between magnets are called magnetic forces.
- The strength of a force is called its magnitude. The greater the magnitude of the magnetic force between two magnets, the more strongly the magnets attract or repel each other.
- You can change the magnitude of a magnetic force between two magnets by changing the distance between them. The magnitude of the magnetic force is greater when there is a smaller distance between the magnets. 

Read the following question, generate the background knowledge as the context information that could be helpful for answering the question.
"""