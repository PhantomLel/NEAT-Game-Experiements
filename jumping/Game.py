import pygame as pg
import random
import neat, time

# simple pygame game that allows user to jump to avoid an obstacle

# initialize pygame
pg.init()
font = pg.font.SysFont(None, 24)

fps = 120


class Obstacle:
    def __init__(self, screen_w, screen_h) -> None:
        self.height = random.randint(10, round(screen_h/1.7))
        self.pos = pg.Rect(0, 0, 20, self.height)
        self.pos.bottomleft = (screen_w, screen_h - 20)
        self.dodged = False

    def draw(self, screen):
        pg.draw.rect(screen, (255, 0, 255), self.pos)


class Game:
    def __init__(self, screen, width, height) -> None:
        self.screen = screen
        self.width = width
        self.height = height
        self.clock = pg.time.Clock()
        self.running = True
        self.vy = 0
        self.gravity = 0.3
        # place player at the bottom of the screen
        self.player = pg.Rect(0, 0, 20, 20)
        self.player.center = (width / 2, height - 40)
        self.floor = pg.Rect(0, height - 20, width, 20)
        self.obstacles: list[Obstacle] = []
        self.dodged = 0

        self.frames = 0

        self.net = None
        self.genome = None

        self.record = False
        self.record_frames = []

    def run(self):
        while self.running:
            self.loop()
            if self.frames == 1500:
                print(self.genome.fitness)
                break

    def train(self, genome, config):
        start = time.time()
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.run()
        return time.time() - start

    def jump(self):
        if self.vy == 0 and self.player.bottom == self.floor.top:
            self.vy = -13

    def draw(self):
        pg.draw.rect(self.screen, (50, 210, 255), self.player)
        # draw floor
        pg.draw.rect(self.screen, (255, 255, 255), self.floor)
        # handle obstacles
        for obstacle in self.obstacles:
            obstacle.pos.x -= 3
            obstacle.draw(self.screen)
            if obstacle.pos.right < 0:
                self.obstacles.remove(obstacle)
            elif self.player.colliderect(obstacle.pos):
                self.genome.fitness -= 200
                self.obstacles.remove(obstacle)
            # check if player dodged obstacle
            elif obstacle.pos.right < self.player.left and not obstacle.dodged:
                self.dodged += 1
                obstacle.dodged = True
                self.genome.fitness += 145

        global fps
        # fps may become too much at times to render
        try:
            frame_rate_render = font.render(
                f"FPS: {round(self.clock.get_fps())}/{fps}", True, (255, 255, 255)
            )
            self.screen.blit(frame_rate_render, (10, 30))
        except:
            pass

    def loop(self):
        global fps
        self.screen.fill((0, 0, 0))
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            # use arrow keys to change fps

            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.running = False
                if event.key == pg.K_RIGHT:
                    fps += 50
                if event.key == pg.K_LEFT:
                    if fps > 50:
                        fps -= 50

        # check for jump
        if self.vy != 0 or self.player.bottom != self.floor.top:
            self.player.y += self.vy
            self.vy += self.gravity
            # check if player is on the floor
            if self.player.bottom > self.floor.top:
                self.player.bottom = self.floor.top
                # reset vy
                self.vy = 0
        else:
            # get decistion from neural network
            self.get_input()

        # if random.randint(0, 100) == 0 and len(self.obstacles) <= 2:
        if len(self.obstacles) < 1:
            self.obstacles.append(Obstacle(self.width, self.height))

        self.draw()

        # updae screen every few frames or if fps is low
        if self.frames % 150 == 0 or fps < 200:
            pg.display.flip()

        if self.record:
            # get the surface as a numpy array and swap axes. Much, much faster than writing an image
            self.record_frames.append(pg.surfarray.array3d(self.screen).swapaxes(0, 1))
        self.frames += 1
        self.clock.tick(fps)

    def get_input(self):
        # find closest obstacle still in front of player
        closest = None
        dist = 0
        for obstacle in self.obstacles:
            if obstacle.pos.left > self.player.right:
                if closest == None or obstacle.pos.left < closest.pos.left:
                    closest = obstacle
                    dist = obstacle.pos.right - self.player.right

        no_obstacle = 0 if closest is None else 1
        height = closest.height if closest is not None else -10000
        # input distance to next obstacle
        output = self.net.activate((dist, no_obstacle, height))
        # get decision from neural network
        dec = output.index(max(output))
        if dec == 1:
            self.jump()
            # discourage random jumping
            self.genome.fitness -= 40

    def test(self, genome, config) -> list:
        global fps
        fps = 100
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.record = True
        self.run()
        print(self.genome.fitness)
        return self.record_frames
