from controller import Robot, Motor, DistanceSensor
import numpy as np
import mlp as ntw
from typing import List

HIDDEN_LAYERS = [12]
MIN_GROUND_SENSOR_VALUE = 280
MAX_GROUND_SENSOR_VALUE = 760
MIN_DISTANCE_SENSOR_VALUE = 80
MAX_DISTANCE_SENSOR_VALUE = 1000
PRINT_EVERY = 1000  # Print every n steps

EPSILON = 0.25  # Small value to determine if the robot is spinning


class Controller:
    def __init__(self, robot: Robot):
        # Robot Parameters
        # Please, do not change these parameters
        self.robot: Robot = robot
        self.time_step = 32  # ms
        self.max_speed = 1  # m/s

        # MLP Parameters and Variables
        ### Define below the architecture of your MLP network.
        ### Add the number of neurons for each layer.
        ### The number of neurons should be in between of 1 to 20.
        ### Number of hidden layers should be one or two.
        self.number_input_layer = (
            13  # 8 proximity + 3 ground sensors + 2 previous motor speeds
        )
        # Example with one hidden layers: self.number_hidden_layer = [5]
        # Example with two hidden layers: self.number_hidden_layer = [7,5]
        self.number_hidden_layer = HIDDEN_LAYERS
        self.number_output_layer = 2

        # Create a list with the number of neurons per layer
        self.number_neuros_per_layer = []
        self.number_neuros_per_layer.append(self.number_input_layer)
        self.number_neuros_per_layer.extend(self.number_hidden_layer)
        self.number_neuros_per_layer.append(self.number_output_layer)

        # Initialize the network
        self.network = ntw.MLP(self.number_neuros_per_layer)
        self.inputs = []

        # Calculate the number of weights of your MLP
        self.number_weights = 0
        for n in range(1, len(self.number_neuros_per_layer)):
            if n == 1:
                # Input + bias
                self.number_weights += (
                    self.number_neuros_per_layer[n - 1] + 1
                ) * self.number_neuros_per_layer[n]
            else:
                self.number_weights += (
                    self.number_neuros_per_layer[n - 1]
                    * self.number_neuros_per_layer[n]
                )

        # Enable Motors
        self.left_motor: Motor = self.robot.getDevice("left wheel motor")
        self.right_motor: Motor = self.robot.getDevice("right wheel motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        # Discovered position sensors ! But they still increase when touching a wall
        # self.left_position_sensor = self.robot.getPositionSensor("left wheel sensor")
        # self.left_position_sensor.enable(self.time_step)
        # self.right_position_sensor = self.robot.getPositionSensor("right wheel sensor")
        # self.right_position_sensor.enable(self.time_step)

        self.velocity_left = 0
        self.velocity_right = 0

        # Enable Proximity Sensors
        self.proximity_sensors: List[DistanceSensor] = []
        for i in range(8):
            sensor_name = "ps" + str(i)
            self.proximity_sensors.append(self.robot.getDevice(sensor_name))
            self.proximity_sensors[i].enable(self.time_step)

        # Enable Ground Sensors
        self.left_ir: DistanceSensor = self.robot.getDevice("gs0")
        self.left_ir.enable(self.time_step)
        self.center_ir: DistanceSensor = self.robot.getDevice("gs1")
        self.center_ir.enable(self.time_step)
        self.right_ir: DistanceSensor = self.robot.getDevice("gs2")
        self.right_ir.enable(self.time_step)

        # Enable Emitter and Receiver (to communicate with the Supervisor)
        self.emitter = self.robot.getDevice("emitter")
        self.receiver = self.robot.getDevice("receiver")
        self.receiver.enable(self.time_step)
        self.receivedData = ""
        self.receivedDataPrevious = ""
        self.flagMessage = False

        # Fitness value (initialization fitness parameters once)
        self.fitness_values = []
        self.fitness = 0

        # Add line tracking
        self.steps_on_white = 0
        self.max_steps_on_white = 0
        self.steps_avoiding = 0
        self.last_velocities: List[tuple[float, float]] = []

        self.print_every = 0

    def check_for_new_genes(self):
        if self.flagMessage:
            # Split the list based on the number of layers of your network
            part = []
            for n in range(1, len(self.number_neuros_per_layer)):
                if n == 1:
                    part.append(
                        (self.number_neuros_per_layer[n - 1] + 1)
                        * (self.number_neuros_per_layer[n])
                    )
                else:
                    part.append(
                        self.number_neuros_per_layer[n - 1]
                        * self.number_neuros_per_layer[n]
                    )

            # Set the weights of the network
            data = []
            weightsPart = []
            sum = 0
            for n in range(1, len(self.number_neuros_per_layer)):
                if n == 1:
                    weightsPart.append(self.receivedData[n - 1 : part[n - 1]])
                elif n == (len(self.number_neuros_per_layer) - 1):
                    weightsPart.append(self.receivedData[sum:])
                else:
                    weightsPart.append(self.receivedData[sum : sum + part[n - 1]])
                sum += part[n - 1]
            for n in range(1, len(self.number_neuros_per_layer)):
                if n == 1:
                    weightsPart[n - 1] = weightsPart[n - 1].reshape(
                        [
                            self.number_neuros_per_layer[n - 1] + 1,
                            self.number_neuros_per_layer[n],
                        ]
                    )
                else:
                    weightsPart[n - 1] = weightsPart[n - 1].reshape(
                        [
                            self.number_neuros_per_layer[n - 1],
                            self.number_neuros_per_layer[n],
                        ]
                    )
                data.append(weightsPart[n - 1])
            self.network.weights = data

            # Reset fitness and line tracking when getting new genes
            self.fitness_values = []
            self.steps_on_white = 0
            self.max_steps_on_white = 0

            self.print_every = 0

    def clip_value(self, value, min_max):
        if value > min_max:
            return min_max
        elif value < -min_max:
            return -min_max
        return value

    def sense_compute_and_actuate(self):
        # MLP:
        #   Input == sensory data
        #   Output == motors commands
        output = self.network.propagate_forward(self.inputs)
        self.velocity_left = output[0]
        self.velocity_right = output[1]

        # Multiply the motor values by 3 to increase the velocities
        self.left_motor.setVelocity(self.velocity_left * 3)
        self.right_motor.setVelocity(self.velocity_right * 3)

    def calculate_fitness(self):
        self.print_every += 1
        printing = False
        if self.print_every >= PRINT_EVERY:
            self.print_every = 0
            printing = True

        ### Define the fitness function to avoid collision
        isAvoiding = False
        avoidCollisionFitness = 0
        # Get front distance sensors values
        # If an obstacle is detected in front of the robot reduce fitness
        obstacle_tolerance = 400
        for ds in self.proximity_sensors:
            ds_value = ds.getValue()
            # Avoid sensor noise
            if ds_value < MAX_DISTANCE_SENSOR_VALUE * 3:
                if ds_value > MIN_DISTANCE_SENSOR_VALUE:
                    isAvoiding = True
                if ds_value > MIN_DISTANCE_SENSOR_VALUE + obstacle_tolerance:
                    # Penalty for being too close to an obstacle
                    too_close_penalty = (
                        ds_value - MIN_DISTANCE_SENSOR_VALUE
                    ) / MAX_DISTANCE_SENSOR_VALUE
                    too_close_penalty /= 4
                    if avoidCollisionFitness < too_close_penalty:
                        avoidCollisionFitness = too_close_penalty

        avoidCollisionFitness = -avoidCollisionFitness

        ### Define the fitness function to increase the speed of the robot and
        ### to encourage the robot to move forward only
        # Get the left and right wheel speeds
        left_speed = self.left_motor.getVelocity()
        right_speed = self.right_motor.getVelocity()
        #todo Not sure about that v_change part, maybe a proper "has_line" variable should fit better (going to false when on white or colliding for too long)
        v_change = 0
        if False:  # disable v_change penalty for now
            self.last_velocities.append((left_speed, right_speed))
            if len(self.last_velocities) > 50:
                self.last_velocities.pop(0)
                v_change = sum(
                    abs(self.last_velocities[i][0] - self.last_velocities[i - 1][0])
                    + abs(self.last_velocities[i][1] - self.last_velocities[i - 1][1])
                    for i in range(-49, 0)
                )
                if v_change < 10:  # almost no change for 40 steps
                    avoidCollisionFitness -= 2

        forwardFitness = (left_speed + right_speed) / (2 * self.max_speed)
        # forwardFitness *= 1.5

        ### Define the fitness function to encourage the robot to follow the line
        # get ground sensors values - 760 is white, 300 is black
        left = self.left_ir.getValue() < 700
        centre = self.center_ir.getValue() < 700
        right = self.right_ir.getValue() < 700

        # Check if robot has lost the line
        if not (left and centre and right) and not isAvoiding:
            self.steps_on_white += 1
            if self.steps_on_white > self.max_steps_on_white:
                self.max_steps_on_white = self.steps_on_white
        else:
            self.steps_on_white = 0

        # Calculate line following fitness with permanent penalty AND progress tracking
        followLineFitness = 0
        followLineFitness = left + right + centre  # Immediate reward

        ### Define the fitness function to avoid spinning behaviour
        spinningFitness = 0
        speed_difference = abs(left_speed - right_speed)
        # Discourage negative correlation between wheel speeds
        if isAvoiding:
            self.steps_avoiding += 1
            # if abs(left_speed - right_speed) < EPSILON:
            #     avoidCollisionFitness -= 3
            if speed_difference > EPSILON * 3:
                spinningFitness += 1
            # Encourage going on white to avoid the obstacle
            if followLineFitness <= 0 and self.steps_avoiding < 300:
                followLineFitness += 2
            else:
                followLineFitness -= 1
        else:
            self.steps_avoiding = 0
            # White penalty
            if followLineFitness <= 1:
                followLineFitness -= 2

            # Discourage large differences between wheel speeds
            if speed_difference > EPSILON:
                spinningFitness -= 1 + speed_difference

            if left_speed < 0 or right_speed < 0:
                spinningFitness = -20

        if self.steps_avoiding > 1000:
            followLineFitness -= 5

        # followLineFitness -= min((self.max_steps_on_white / 120) ** 2, 10)  # Penalty for time on white

        # # define the back LR sensors
        # backleft = self.proximity_sensors[3].getValue()
        # backright = self.proximity_sensors[4].getValue()
        # # punish harsh when obstacle detected in rear sensors
        # if backleft > MIN_DISTANCE_SENSOR_VALUE or backright > MIN_DISTANCE_SENSOR_VALUE:
        #     forwardFitness -= 1

        ### Encourage exploration

        ### Define the fitness function of this iteration which should be a combination of the previous functions
        combinedFitness = (
            forwardFitness + followLineFitness + avoidCollisionFitness + spinningFitness
        )

        self.fitness_values.append(combinedFitness)
        self.fitness = np.mean(self.fitness_values)

        if printing:
            print(
                "Fitness: {:.1f}, Forward: {:.1f}, Follow Line: {:.1f}, Avoid Collision: {:.1f}, Spinning: {:.1f}, MaxWhite: {}, Avoid: {}, v_change: {:.1f}".format(
                    self.fitness,
                    forwardFitness,
                    followLineFitness,
                    avoidCollisionFitness,
                    spinningFitness,
                    self.max_steps_on_white,
                    self.steps_avoiding,
                    v_change,
                )
            )

    def handle_emitter(self):
        # Send the self.fitness value to the supervisor
        data = str(self.number_weights)
        data = "weights: " + data
        string_message = str(data)
        string_message = string_message.encode("utf-8")
        # print("Robot send:", string_message)
        self.emitter.send(string_message)

        # Send the self.fitness value to the supervisor
        data = str(self.fitness)
        data = "fitness: " + data
        string_message = str(data)
        string_message = string_message.encode("utf-8")
        # print("Robot send fitness:", string_message)
        self.emitter.send(string_message)

    def handle_receiver(self):
        if self.receiver.getQueueLength() > 0:
            while self.receiver.getQueueLength() > 0:
                # Adjust the Data to our model
                # Webots 2022:
                # self.receivedData = self.receiver.getData().decode("utf-8")
                # Webots 2023:
                self.receivedData = self.receiver.getString()

                self.receivedData = self.receivedData[1:-1]
                self.receivedData = self.receivedData.split()
                x = np.array(self.receivedData)
                self.receivedData = x.astype(float)
                # print("Controller handle receiver data:", self.receivedData)
                self.receiver.nextPacket()

            # Is it a new Genotype?
            if not np.array_equal(self.receivedDataPrevious, self.receivedData):
                self.flagMessage = True

            else:
                self.flagMessage = False

            self.receivedDataPrevious = self.receivedData
        else:
            # print("Controller receiver q is empty")
            self.flagMessage = False

    def run_robot(self):
        # Main Loop
        while self.robot.step(self.time_step) != -1:
            # This is used to store the current input data from the sensors
            self.inputs = []

            # Emitter and Receiver
            # Check if there are messages to be sent or read to/from our Supervisor
            self.handle_emitter()
            self.handle_receiver()

            # Read Ground Sensors
            left = self.left_ir.getValue()
            center = self.center_ir.getValue()
            right = self.right_ir.getValue()
            # print("Ground Sensors \n    left {} center {} right {}".format(left,center,right))

            ### Please adjust the ground sensors values to facilitate learning
            min_gs = MIN_GROUND_SENSOR_VALUE
            max_gs = MAX_GROUND_SENSOR_VALUE

            if left > max_gs:
                left = max_gs
            if center > max_gs:
                center = max_gs
            if right > max_gs:
                right = max_gs
            if left < min_gs:
                left = min_gs
            if center < min_gs:
                center = min_gs
            if right < min_gs:
                right = min_gs

            # Normalize the values between 0 and 1 and save data
            self.inputs.append((left - min_gs) / (max_gs - min_gs))
            self.inputs.append((center - min_gs) / (max_gs - min_gs))
            self.inputs.append((right - min_gs) / (max_gs - min_gs))
            # print("Ground Sensors \n    left {} center {} right {}".format(self.inputs[0],self.inputs[1],self.inputs[2]))

            # Read Distance Sensors
            for i in range(8):
                ### Select the distance sensors that you will use
                if (
                    i == 0
                    or i == 1
                    or i == 2
                    or i == 3
                    or i == 4
                    or i == 5
                    or i == 6
                    or i == 7
                ):
                    temp = self.proximity_sensors[i].getValue()

                    ### Please adjust the distance sensors values to facilitate learning
                    min_ds = MIN_DISTANCE_SENSOR_VALUE
                    max_ds = MAX_DISTANCE_SENSOR_VALUE

                    if temp > max_ds:
                        temp = max_ds
                    if temp < min_ds:
                        temp = min_ds

                    # Normalize the values between 0 and 1 and save data
                    self.inputs.append(max(0, (temp - min_ds) / (max_ds - min_ds)))
                    # print("Distance Sensors - Index: {}  Value: {}".format(i,self.proximity_sensors[i].getValue()))

            self.inputs.append(self.clip_value(self.velocity_left, 1))
            self.inputs.append(self.clip_value(self.velocity_right, 1))

            # GA Iteration
            # Verify if there is a new genotype to be used that was sent from Supervisor
            self.check_for_new_genes()
            # Define the robot's actuation (motor values) based on the output of the MLP
            self.sense_compute_and_actuate()
            # Calculate the fitnes value of the current iteration
            self.calculate_fitness()

            # End of the iteration


if __name__ == "__main__":
    # Call Robot function to initialize the robot
    my_robot = Robot()
    # Initialize the parameters of the controller by sending my_robot
    controller = Controller(my_robot)
    # Run the controller
    controller.run_robot()
