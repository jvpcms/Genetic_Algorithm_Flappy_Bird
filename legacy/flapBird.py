from single_processing_version.gameClasses import *

# screen size, colors
resolution = (900, 600)
background = (200, 200, 200)


def simulate(networks, number_survivors):

    current_loop, perfomance, survivors = 0, [], []
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

    while True:

        current_loop += 1

        # check if wall is outside of screen
        for wall in walls:
            if wall.x < -wall.thickness:
                walls[0] = walls[1]
                walls[1] = Wall(resolution)
                current_wall = 0
            wall.update()

        if current_wall == 0 and walls[current_wall].x + walls[current_wall].thickness < birds[0].x:
            current_wall = 1

        for b in range(population_size):

            if not alive[b]:
                continue

            birds[b].update()

            # collision with walls and ground
            collision = (birds[b].collider.colliderect(walls[0].collider_up) or
                         birds[b].collider.colliderect(walls[0].collider_down) or
                         birds[b].y + birds[b].size > resolution[1])

            # change state
            if collision:

                alive[b] = 0
                n_live -= 1

                if n_live < number_survivors:
                    survivors.append(networks[b])
                    perfomance.append(current_loop)
                if n_live == 0:
                    return survivors, perfomance

            # input parametres for neural network
            enter_parametres = [(birds[b].y - walls[current_wall].y) / resolution[1],
                                (walls[current_wall].x - birds[b].x) / resolution[0]]

            # value of output nodes (one node in thiss case)
            activation = networks[b].predict(enter_parametres)

            if activation >= 0.5:
                birds[b].jump()
