import pygame
import random

gravity = 0.001


class Bird:

    def __init__(self, resolution):
        self.x = resolution[0] / 4
        self.y = resolution[1] / 2
        self.color = (50, 50, random.randint(0, 255))
        self.size = int(resolution[0] / 90)

        self.speed = 0
        self.collider = pygame.Rect(self.x, self.y, self.size, self.size)

    def display(self, scr):
        pygame.draw.rect(scr, self.color, (self.x, self.y, self.size, self.size))

    def update(self):
        if self.y < 0:
            self.y = 0
            self.speed = 0
        if self.speed > 0.5:
            self.speed = 0.5

        self.speed += gravity
        self.y += self.speed
        self.collider.y = self.y

    def jump(self):
        self.speed = -0.5


class Wall:

    def __init__(self, res):
        self.x = res[0]
        self.y = random.randint(0, int(res[1] / 2)) + (res[1] / 4)
        self.thickness = int(res[0] / 45)
        self.color = (25, 25, 25)
        self.gap = int(8 * res[0] / 90)

        self.lengh_bottom = res[1] - self.y - self.gap

        self.speed = 0.3
        self.collider_up = pygame.Rect(self.x, 0, self.thickness, self.y)
        self.collider_down = pygame.Rect(self.x, self.y + self.gap, self.thickness, self.lengh_bottom)

    def display(self, scr):
        pygame.draw.rect(scr, self.color, (self.x, 0, self.thickness, self.y))
        pygame.draw.rect(scr, self.color, (self.x, self.y + self.gap, self.thickness, self.lengh_bottom))

    def update(self):
        self.x -= self.speed
        self.collider_up.x = self.x
        self.collider_down.x = self.x
