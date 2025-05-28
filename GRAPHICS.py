import random
import pygame
import json
import numpy as np
import matplotlib.pyplot as plt


num_floors = 4
time_per_floor = 3
time_per_stop = 5
capacity = 5
initial_passengers = 15
NUM_EPISODES = 1000000




class ElevatorVisualizer:
    def __init__(self, env, num_floors=num_floors):
        pygame.init()
        self.env = env
        self.width = 800
        self.height = 650
        self.floor_height = self.height // num_floors
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Elevator Simulation")
        self.font = pygame.font.SysFont(None, 20, bold=True)
        self.clock = pygame.time.Clock()
        self.simulation_done = False

    def draw(self):
        self.screen.fill((240, 240, 240))

        # Draw floors
        for i in range(num_floors):
            y = self.height - (i + 1) * self.floor_height
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.width, y), 2)
            label = self.font.render(f"Floor {i}", True, (50, 50, 50))
            self.screen.blit(label, (10, y + 5))

        # Draw elevator
        elevator = self.env.building.elevator
        y = self.height - (elevator.current_floor + 1) * self.floor_height
        elevator_w, elevator_h = 200, self.floor_height - 20
        elevator_x = self.width // 2 - elevator_w // 2
        elevator_rect = pygame.Rect(elevator_x, y + 10, elevator_w, elevator_h)
        pygame.draw.rect(self.screen, (0, 150, 250), elevator_rect, border_radius=10)

        # Passengers inside the elevator
        for idx, p in enumerate(elevator.passengers):
            px = elevator_rect.x + 20 + (idx % 4) * 45
            py = elevator_rect.y + 20 + (idx // 4) * 45
            pygame.draw.circle(self.screen, (255, 255, 255), (px, py), 25)
            pygame.draw.circle(self.screen, (0, 0, 0), (px, py), 20, 2)
            text = self.font.render(f"{p.start_floor} -> {p.destination_floor}", True, (0, 0, 0))
            text_rect = text.get_rect(center=(px, py))
            self.screen.blit(text, text_rect)

        # Waiting passengers
        for floor in self.env.building.floors:
            fy = self.height - (floor.floor_number + 1) * self.floor_height
            for idx, p in enumerate(floor.waiting_passengers):
                px = 40 + idx * 55
                py = fy + self.floor_height // 2
                pygame.draw.circle(self.screen, (255, 255, 255), (px, py), 25)
                pygame.draw.circle(self.screen, (0, 0, 0), (px, py), 20, 2)
                text = self.font.render(f"{p.start_floor} -> {p.destination_floor}", True, (0, 0, 0))
                text_rect = text.get_rect(center=(px, py))
                self.screen.blit(text, text_rect)

        # Dropped off passengers (optional: last 5 only)
        if hasattr(elevator, 'dropped_last_step'):
            for idx, p in enumerate(elevator.dropped_last_step[-5:]):
                px = elevator_rect.right + 40 + (idx % 3) * 60
                py = elevator_rect.y + 30 + (idx // 3) * 50
                pygame.draw.circle(self.screen, (255, 255, 255), (px, py), 25)
                pygame.draw.circle(self.screen, (150, 0, 0), (px, py), 20, 2)
                text = self.font.render(f"{p.start_floor} -> {p.destination_floor}", True, (0, 0, 0))
                text_rect = text.get_rect(center=(px, py))
                self.screen.blit(text, text_rect)

        # Info bar
        info_text = self.font.render(f"Time: {self.env.time} | Floor: {elevator.current_floor}", True, (0, 0, 0))
        self.screen.blit(info_text, (self.width - info_text.get_width() - 20, 10))

        if self.simulation_done:
            msg = self.font.render("Simulation finished. Press any key.", True, (200, 0, 0))
            msg_rect = msg.get_rect(center=(self.width // 2, self.height - 30))
            self.screen.blit(msg, msg_rect)

        pygame.display.flip()
        self.clock.tick(2)

    def wait_for_exit(self):
        self.simulation_done = True
        while True:
            self.draw()
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                    pygame.quit()
                    exit()

class Passenger:
    def __init__(self, start_floor, destination_floor, simulation_time):
        self.start_floor = start_floor
        self.destination_floor = destination_floor
        self.request_time = simulation_time
        self.boarding_time = None
        self.dropoff_time = None

class Floor:
    def __init__(self, floor_number):
        self.floor_number = floor_number
        self.waiting_passengers = []

    def add_passenger(self, passenger):
        self.waiting_passengers.append(passenger)

class Elevator:
    def __init__(self, building, capacity=capacity):
        self.building = building
        self.current_floor = 0
        self.passengers = []
        self.time_until_next_action = 0
        self.total_floor_wait_time = 0
        self.total_elevator_time = 0
        self.capacity = capacity
        self.last_direction = 1
        self.dropped_last_step = []  # Store dropped passengers for the last step

    def move(self, action):
        if self.time_until_next_action > 0:
            self.time_until_next_action -= 1
            return

        if action == 1 and self.current_floor < num_floors - 1:
            self.current_floor += 1
            self.building.env.time += time_per_floor
            self.time_until_next_action = time_per_floor
            self.last_direction = 1

        elif action == -1 and self.current_floor > 0:
            self.current_floor -= 1
            self.building.env.time += time_per_floor
            self.time_until_next_action = time_per_floor
            self.last_direction = -1

        elif action == 0:
            self.drop_off_passengers()
            self.board_passengers(self.building.floors[self.current_floor])
            self.building.env.time += time_per_stop
            self.time_until_next_action = time_per_stop
            self.last_direction = 0

    def drop_off_passengers(self):
        dropped = [p for p in self.passengers if p.destination_floor == self.current_floor]
        for p in dropped:
            p.dropoff_time = self.building.env.time
            self.total_elevator_time += p.dropoff_time - p.boarding_time
            self.passengers.remove(p)
        self.dropped_last_step = dropped.copy()
        return len(dropped)

    def board_passengers(self, floor):
        boarded_count = 0
        while len(self.passengers) < self.capacity and floor.waiting_passengers:
            p = floor.waiting_passengers.pop(0)
            p.boarding_time = self.building.env.time
            self.total_floor_wait_time += p.boarding_time - p.request_time
            self.passengers.append(p)
            boarded_count += 1
        return boarded_count

class Building:
    def __init__(self, env, num_floors=num_floors):
        self.env = env
        self.floors = [Floor(i) for i in range(num_floors)]
        self.elevator = Elevator(self)

    def generate_passenger(self):
        start, dest = random.sample(range(num_floors), 2)
        passenger = Passenger(start, dest, self.env.time)
        self.floors[start].add_passenger(passenger)

    def total_waiting(self):
        return sum(len(f.waiting_passengers) for f in self.floors)

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        q_values = [self.q_table.get((state, a), 0.0) for a in self.actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table.get((state, action), 0.0)
        future_q = max([self.q_table.get((next_state, a), 0.0) for a in self.actions])
        updated_q = current_q + self.alpha * (reward + self.gamma * future_q - current_q)
        self.q_table[(state, action)] = updated_q


    def save_model(self, filename):
        q_serializable = {str(k): v for k, v in self.q_table.items()}

        q_values_json = json.dumps(q_serializable)
        ModelData_json = json.dumps(results)
        with open(f'{filename}.json', 'w') as file:
            file.write(ModelData_json + '\n')
            file.write(q_values_json + '\n')
        print("The trained model is saved successfully!")
      
       

    def load_model(self, filename):
        with open(f'{filename}.json', 'r') as file:
            results_loaded  = json.loads(file.readline())
            q_values = json.loads(file.readline())       
        self.q_table = {eval(k): v for k, v in q_values.items()}       
        return results_loaded
 
    
   
        # Convert non-serializable keys into strings

class ElevatorEnv:
    def __init__(self):
        self.time = 0
        self.building = Building(self)
        self.agent = QLearningAgent(actions=[-1, 0, 1])

    def get_state(self):
        elevator = self.building.elevator
        current_floor = elevator.current_floor
        direction = elevator.last_direction

        passenger_destinations = [-1] * elevator.capacity
        for i, p in enumerate(elevator.passengers):
            passenger_destinations[i] = p.destination_floor

        waiting_requests = [1 if floor.waiting_passengers else 0 for floor in self.building.floors]

        return current_floor, direction, tuple(passenger_destinations), tuple(waiting_requests)

    def done(self):
        return (self.building.total_waiting() == 0 and
                len(self.building.elevator.passengers) == 0)

    def step(self):
        state = self.get_state()
        action = self.agent.choose_action(state)

        self.building.elevator.move(action)

        dropped = self.building.elevator.drop_off_passengers()
        boarded = self.building.elevator.board_passengers(
            self.building.floors[self.building.elevator.current_floor]
        )

        reward = self.get_reward(dropped, boarded)
        next_state = self.get_state()
        self.agent.update_q_table(state, action, reward, next_state)
        return reward, boarded, dropped

    def get_reward(self, dropped, boarded):
        waiting = self.building.total_waiting()
        return 10 * dropped + 2 * boarded - 1 - 2 * waiting
    
    
    def run(self, episode_number=0, learn=True, visualizer=None, initial_passengers=initial_passengers):
        ran_initial_passengers = random.randint(5, initial_passengers)
        for _ in range(ran_initial_passengers):
            self.building.generate_passenger()

        total_boarded = total_dropped = total_reward = 0
        self.time = 0
        passenger_generated = ran_initial_passengers

        while not self.done():
            state = self.get_state()
            action = self.agent.choose_action(state)
            self.building.elevator.move(action)

            dropped = self.building.elevator.drop_off_passengers()
            boarded = self.building.elevator.board_passengers(
                self.building.floors[self.building.elevator.current_floor]
            )

            reward = self.get_reward(dropped, boarded)
            next_state = self.get_state()

            if learn:
                self.agent.update_q_table(state, action, reward, next_state)

            total_boarded += boarded
            total_dropped += dropped
            total_reward += reward

            if random.random() < 0.2 and passenger_generated < 49:
                self.building.generate_passenger()
                self.building.generate_passenger()
                passenger_generated += 2

            self.time += 1

            if visualizer:
                visualizer.draw()
                for event in pygame.event.get():  # 
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()

        avg_wait = self.building.elevator.total_floor_wait_time / total_dropped if total_dropped else 0
        avg_elevator = self.building.elevator.total_elevator_time / total_dropped if total_dropped else 0
        if episode_number % 500000 == 0:
            print(f"\n--- Episode {episode_number} ---")
            print(f"Time elapsed         : {self.time}")
            print(f"Passengers boarded   : {total_boarded}")
            print(f"Passengers dropped   : {total_dropped}")
            print(f"Average wait time    : {avg_wait:.2f}")
            print(f"Average ride time    : {avg_elevator:.2f}")
            print(f"Total reward         : {total_reward:.2f}")

        return {
            'episode': episode_number,
            'reward': total_reward,
            'boarded': total_boarded,
            'dropped': total_dropped,
            'avg_wait': avg_wait,
            'avg_ride': avg_elevator,
            'time': self.time
        }





if __name__ == "__main__":
    agent = QLearningAgent(actions=[-1, 0, 1])
    results = []

    #  Load or Train
    use_trained = input("📂 Load a trained elevator model? (y/n): ").strip().lower()
    if use_trained == 'y':
        file_name = input("Enter filename to load (e.g. elevator_model): ").strip()
        results = agent.load_model(file_name)
        skip_training = True
    else:
        skip_training = False

    #  Train if needed
    if not skip_training:
        print("\nTraining started...")
        for ep in range(1, NUM_EPISODES + 1):
            env = ElevatorEnv()
            env.agent = agent
            stats = env.run(episode_number=ep, learn=True, initial_passengers=initial_passengers)
            results.append(stats)
        print("\n Training finished!")

        # 💾 Save the model
        save_input = input("\n Do you want to save the trained model? (y/n): ").strip().lower()
        if save_input == 'y':
            file_name = input("Enter filename to save (e.g. elevator_model.json): ").strip()
            agent.save_model(file_name)

           #

    # 📊 Show summary if available
    show_plot = input("📂 Load a plot and res? (y/n): ").strip().lower()
    if show_plot == 'y':

        if results:
            final = results[-1]
            print(f"\n Final Training Results:")
            print(f"Episode: {final['episode']}, Reward: {final['reward']:.2f}")
            print(f"Dropped: {final['dropped']}, Boarded: {final['boarded']}, Avg Wait: {final['avg_wait']:.2f}, Avg Ride: {final['avg_ride']:.2f}")

        # 📈 Plot results if available
        if results:
            import matplotlib.pyplot as plt

            def moving_average(data, window_size=100):
                return [
                    sum(data[max(0, i - window_size + 1):i + 1]) / (i - max(0, i - window_size + 1) + 1)
                    for i in range(len(data))
                ]

            episodes = [r['episode'] for r in results]
            rewards = [r['reward'] for r in results]
            avg_waits = [r['avg_wait'] for r in results]
            avg_rides = [r['avg_ride'] for r in results]

            # Smooth the data
            rewards_smooth = moving_average(rewards)
            avg_waits_smooth = moving_average(avg_waits)
            avg_rides_smooth = moving_average(avg_rides)

            # Plotting
            plt.figure(figsize=(14, 6))

            plt.subplot(1, 3, 1)
            plt.plot(episodes, rewards_smooth)
            plt.title("Smoothed Total Reward")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.grid(True)

            plt.subplot(1, 3, 2)
            plt.plot(episodes, avg_waits_smooth, color='orange')
            plt.title("Smoothed Avg Wait Time")
            plt.xlabel("Episode")
            plt.ylabel("Time")
            plt.grid(True)

            plt.subplot(1, 3, 3)
            plt.plot(episodes, avg_rides_smooth, color='green')
            plt.title("Smoothed Avg Ride Time")
            plt.xlabel("Episode")
            plt.ylabel("Time")
            plt.grid(True)

            plt.tight_layout()
            plt.show()

    # 👁 Visualize one episode
    visualize = input("\n Do you want to visualize one episode with the trained agent? (y/n): ").strip().lower()
    if visualize == 'y':
        print("Starting visualized episode...")
        env = ElevatorEnv()
        env.agent = agent
        env.agent.epsilon = 0.0
        visualizer = ElevatorVisualizer(env)
        env.run(learn=False, visualizer=visualizer, initial_passengers=10)
        visualizer.wait_for_exit()
    else:
        print("Exiting. Have a great day! ")
