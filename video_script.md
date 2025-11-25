# Video Presentation Requirements
Your group must submit a 10-minute video showcasing your work. The format is:
Structure:
Total Duration: 10 minutes
Group Size: 4 students
Presentation Breakdown:
4 minutes: Two students present the BBR approach.
4 minutes: The other two students present the ER approach.
2 minutes: Two students (one from the BBR and one from the ER approach) discuss
the contrasts and similarities between the two approaches.
Participation
Every student in the group must speak during the video. You may use slides, screen
recordings of simulations, your poster, or annotated visuals to support your
explanation.


# Genetic based training of e-puck to follow a line and avoid obstacles

## Main fitness components

- Following the line (ground sensor on black)
- Moving forward (wheel spinning faster is better)
- Avoiding obstacles (using distance sensors)
- Avoiding spinning in place (wheels turning at different speeds)

##

Here's e-puck, a small mobile robot equipped with ground sensors to detect lines and distance sensors to sense obstacles.
Today, we're going to see how different approaches can be use to create a controller for that robot.

### BBR

The first method is Behaviour-based Robotics, or BBR. The idea is to design and implement simple behaviours that can be combined to achieve complex tasks.
For example, we can create a behaviour for following the line, another for avoiding obstacles, and one to join back the line. Each behaviour will generate motor commands based on the sensor inputs.

### ER

#### bullet points

- Arena (line, obstacles, wall)
- ER in general (when and how to use it)
  - 
- 

#### Script

The arena is a square composed of a black line on a white surface with walls at the borders. 3 different obstacles are placed on the line to make the task more challenging.
We used an evolutionary algorithms to automatically generate robot controllers. Instead of manually designing behaviours, we define a fitness function that evaluates how well a controller performs the desired task. The evolutionary algorithm then iteratively improves the controllers based on their fitness scores.
In our case, this is called Evolutionary Robotics, or ER. It has shown great results

The first step was to understand the code given to us
We then experimented with different fitness functions.
