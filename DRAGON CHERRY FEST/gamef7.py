import gym
from gym import spaces
import numpy as np
import pygame
import random
import time
from collections import deque

class PacmanEnv(gym.Env):
    def __init__(self):
        super(PacmanEnv, self).__init__()
        self.grid_size = (20, 20)  # Grid size
        self.screen_size = 600  # Screen size
        self.cell_size = self.screen_size // self.grid_size[0]  # Calculate cell size based on grid size

        # Initialize game attributes
        self.hungry_dragon = [10, 10]  # Start Pac-Man at the center
        self.food_pos = [random.randint(0, 19), random.randint(0, 19)]  # Random food position
        self.special_food_pos = None  # Special food position
        self.enemy_positions = [[random.randint(0, 19), random.randint(0, 19)] for _ in range(6)]  # Six random enemy positions
        self.done = False
        self.score = 0
        self.penalty_score = 0
        self.steps = 0
        self.max_steps = 20

        self.action_space = spaces.Discrete(4)  # Four possible actions
        self.observation_space = spaces.Box(low=0, high=4, shape=(20, 20), dtype=np.float32)

        # Initialize Pygame and sounds
        pygame.mixer.init()
        self.cherry_sound = pygame.mixer.Sound("cherry_sound.wav")
        self.special_food_sound = pygame.mixer.Sound("dragon_hurt.mp3")
        self.enemy_eaten_sound = pygame.mixer.Sound("dragon_hurt.mp3")
        self.enemy_eat_dragon_sound = pygame.mixer.Sound("ha-ha.mp3")
        pygame.mixer.music.load("Hitman.mp3")
        pygame.mixer.music.play(-1)  # Infinite loop

        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption('Dragon Cherry Fest')

    def reset(self):
        self.hungry_dragon = [10, 10]
        self.food_pos = [random.randint(0, 19), random.randint(0, 19)]
        self.special_food_pos = None
        self.enemy_positions = [[random.randint(0, 19), random.randint(0, 19)] for _ in range(6)]
        self.done = False
        self.score = 0
        self.penalty_score = 0
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros(self.grid_size)
        obs[self.hungry_dragon[0], self.hungry_dragon[1]] = 1
        obs[self.food_pos[0], self.food_pos[1]] = 2
        if self.special_food_pos:
            obs[self.special_food_pos[0], self.special_food_pos[1]] = 3
        for pos in self.enemy_positions:
            obs[pos[0], pos[1]] = 4
        return obs

    def step(self, action):
        prev_food_pos = self.food_pos.copy()
        prev_special_food_pos = self.special_food_pos.copy() if self.special_food_pos else None

        if action == 0:  # Up
            self.hungry_dragon[0] = max(0, self.hungry_dragon[0] - 1)
        elif action == 1:  # Down
            self.hungry_dragon[0] = min(self.grid_size[0] - 1, self.hungry_dragon[0] + 1)
        elif action == 2:  # Left
            self.hungry_dragon[1] = max(0, self.hungry_dragon[1] - 1)
        elif action == 3:  # Right
            self.hungry_dragon[1] = min(self.grid_size[1] - 1, self.hungry_dragon[1] + 1)

        reward = 0
        if self.hungry_dragon == prev_food_pos:
            reward = 10
            self.food_pos = [random.randint(0, 19), random.randint(0, 19)]
            self.score += 1
            self.steps += 1
            self.cherry_sound.play()
            if self.score % 4 == 0:
                self.special_food_pos = [random.randint(0, 19), random.randint(0, 19)]

        if self.special_food_pos and self.hungry_dragon == self.special_food_pos:
            reward = 2
            self.special_food_pos = None
            self.score += 2
            self.special_food_sound.play()

        for enemy_pos in self.enemy_positions:
            if self.hungry_dragon == enemy_pos:
                reward = -5
                self.penalty_score += 1
                self.steps += 1
                self.enemy_eaten_sound.play()
                self.enemy_eat_dragon_sound.play()

        if (self.score + self.penalty_score) >= 20:
            self.done = True

        return self._get_obs(), reward, self.done, {}

    def render(self, mode='human'):
        DARK_BLUE = (0, 0, 128)
        WHITE = (255, 255, 255)
        GREY = (192, 192, 192)
        GREEN = (0, 255, 0)
        RED = (255, 0, 0)
        BLACK = (0, 0, 0)

        self.screen.fill(DARK_BLUE)

        # Draw the grid with alternating white and grey squares
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                color = WHITE if (x + y) % 2 == 0 else GREY
                pygame.draw.rect(self.screen, color, pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size))

        # Draw the grid lines
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                pygame.draw.rect(self.screen, BLACK, pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size), 1)

        # Load and scale images
        pacman_image = pygame.image.load("dragon.png")
        cherry_image = pygame.image.load("cherry.png")
        special_food_image = pygame.image.load("special_food.png")
        enemy_image = pygame.image.load("skeleton.png")

        pacman_image = pygame.transform.scale(pacman_image, (self.cell_size, self.cell_size))
        cherry_image = pygame.transform.scale(cherry_image, (self.cell_size, self.cell_size))
        special_food_image = pygame.transform.scale(special_food_image, (self.cell_size, self.cell_size))
        enemy_image = pygame.transform.scale(enemy_image, (self.cell_size, self.cell_size))

        # Blit images
        self.screen.blit(pacman_image, (self.hungry_dragon[1] * self.cell_size, self.hungry_dragon[0] * self.cell_size))
        self.screen.blit(cherry_image, (self.food_pos[1] * self.cell_size, self.food_pos[0] * self.cell_size))
        if self.special_food_pos:
            self.screen.blit(special_food_image, (self.special_food_pos[1] * self.cell_size, self.special_food_pos[0] * self.cell_size))
        for enemy_pos in self.enemy_positions:
            self.screen.blit(enemy_image, (enemy_pos[1] * self.cell_size, enemy_pos[0] * self.cell_size))

        # Display score and penalty bars
        self._display_score_bar(GREEN, BLACK, 10, 10, self.score, self.max_steps, "Score")
        self._display_score_bar(RED, BLACK, 10, 40, self.penalty_score, self.max_steps, "Penalty")

        if self.done:
            self._display_end_message()

        pygame.display.flip()

    def _display_score_bar(self, fill_color, border_color, x, y, score, max_score, label):
        bar_width = 100
        bar_height = 20
        percentage = min(score / max_score, 1.0)
        filled_width = int(percentage * bar_width)
        pygame.draw.rect(self.screen, fill_color, pygame.Rect(x, y, filled_width, bar_height))
        pygame.draw.rect(self.screen, border_color, pygame.Rect(x, y, bar_width, bar_height), 2)
        font = pygame.font.Font(None, 24)
        label_text = font.render(f"{label}", True, border_color)
        self.screen.blit(label_text, (x - 70, y + 2))  # Adjusted position
        score_text = font.render(f"{score}", True, border_color)
        self.screen.blit(score_text, (x + bar_width + 10, y + 2))

    def _display_end_message(self):
        message_font = pygame.font.Font(None, 48)
        if self.score > self.penalty_score:
            message_text = message_font.render("You win!", True, (0, 255, 0))
        elif self.score < self.penalty_score:
            message_text = message_font.render("You lose!", True, (255, 0, 0))
        else:
            message_text = message_font.render("It's a draw!", True, (255, 255, 0))
        self.screen.blit(message_text, (self.screen_size // 2 - 100, self.screen_size // 2 - 50))

    def bfs(self, start, goal):
        queue = deque([(start, [])])
        visited = set()
        while queue:
            (current, path) = queue.popleft()
            if current == goal:
                return path
            if current in visited:
                continue
            visited.add(current)
            x, y = current
            neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            valid_neighbors = [n for n in neighbors if 0 <= n[0] < self.grid_size[0] and 0 <= n[1] < self.grid_size[1]]
            for neighbor in valid_neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
        return []

    def move_enemies(self):
        for i, enemy_pos in enumerate(self.enemy_positions):
            target = (
                self.hungry_dragon[0] + random.randint(-1, 1),
                self.hungry_dragon[1] + random.randint(-1, 1)
            )
            path = self.bfs(tuple(enemy_pos), target)
            if path:
                self.enemy_positions[i] = list(path[0])
            if self.enemy_positions[i] == self.hungry_dragon:
                self.penalty_score += 1
                self.steps += 1



def run_game():
    env = PacmanEnv()
    clock = pygame.time.Clock()
    running = True
    enemy_move_counter = 0
    end_message_displayed = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    env.step(0)
                elif event.key == pygame.K_DOWN:
                    env.step(1)
                elif event.key == pygame.K_LEFT:
                    env.step(2)
                elif event.key == pygame.K_RIGHT:
                    env.step(3)

        if enemy_move_counter % 10 == 0:  # Enemies move twice as fast
            env.move_enemies()

        if (env.score + env.penalty_score) >= 20 and not end_message_displayed:
            env.done = True
            end_message_displayed = True
            env.render()
            pygame.display.flip()
            time.sleep(2)  # Wait for 2 seconds
            running = False

        enemy_move_counter += 1
        env.render()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    run_game()
