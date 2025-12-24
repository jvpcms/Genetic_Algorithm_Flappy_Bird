import sys

from gameClasses import *
from neuralNetwork import *

# screen size, colors
resolution = (900, 600)
background = (200, 200, 200)

networks = load_population('population.json')
population_size = len(networks)

# list of bool indicating bird state, n° of live birds
alive = [1 for _ in range(population_size)]
n_live = population_size

# list of wall elements, there are only 2 at a time
walls = [Wall(resolution) for _ in range(2)]
walls[1].x += resolution[0] / 2
current_wall = 0

# population of birds
birds = [Bird(resolution) for _ in range(population_size)]

# setup screen
pygame.display.init()
screen = pygame.display.set_mode(resolution)

while True:

    screen.fill(background)
    pygame.display.set_caption(f'Ailive Entities: {n_live}')

    # check if wall is outside of screen
    for wall in walls:
        if wall.x < -wall.thickness:
            walls[0] = walls[1]
            walls[1] = Wall(resolution)
            current_wall = 0
        wall.display(screen)
        wall.update()

    if current_wall == 0 and walls[current_wall].x + walls[current_wall].thickness < birds[0].x:
        current_wall = 1

    for b in range(population_size):

        if not alive[b]:
            continue

        birds[b].display(screen)
        birds[b].update()

        # collision with walls and ground
        collision = (birds[b].collider.colliderect(walls[0].collider_up) or
                     birds[b].collider.colliderect(walls[0].collider_down) or
                     birds[b].y + birds[b].size > resolution[1])

        # change state
        if collision:

            alive[b] = 0
            n_live -= 1

            if n_live <= 0:
                pygame.quit()
                sys.exit()

        # input parametres for neural network
        enter_parametres = [birds[b].speed,
                            (birds[b].y - walls[current_wall].y) / resolution[1],
                            (walls[current_wall].x - birds[b].x) / resolution[0]]

        # value of output nodes (one node in thiss case)
        activation = networks[b].predict(enter_parametres)

        if activation >= 0.5:
            birds[b].jump()

    pygame.display.flip()
