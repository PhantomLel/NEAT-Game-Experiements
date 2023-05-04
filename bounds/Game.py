import math
import pygame as pg
import random
import neat, time


pg.init()
font = pg.font.SysFont(None, 24)

fps = 120

class Game:
    def __init__(self, screen, width, height) -> None:
        self.screen = screen
        self.width = width
        self.height = height
        self.clock = pg.time.Clock()
        self.running = True
        # place player at the bottom of the screen
        self.player = pg.Rect(0, 0, 20, 20)
        self.player.center = (width / 2, height/2)
        self.speed = 5# max speed

        self.target = pg.Rect(0, 0, 10, 10)

        self.frames = 0

        self.net = None
        self.genome = None

        self.record = False
        self.follow_mouse = False
        self.record_frames = []

    def run(self):
        while self.running:
            self.loop()
            if self.frames == 1000:
                print(self.genome.fitness)
                break

    def train(self, genome, config):
        start = time.time()
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.run()
        return time.time() - start

    def draw(self):

        if self.follow_mouse and pg.mouse.get_focused():
            self.target.center = pg.mouse.get_pos()
        elif self.frames % 200 == 0:
            self.target.center = (random.randint(30, self.width-30), random.randint(30, self.height-30))
        # draw target
        pg.draw.rect(self.screen, (255, 255, 255), self.target)
        pg.draw.rect(self.screen, (50, 120, 255), self.player)

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

        self.get_input()
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
        # get dx and dy from target to player
        dist = pg.math.Vector2(self.player.center) - pg.math.Vector2(self.target.center)
        # feed inputs into the neural network
        output = self.net.activate((dist.x, dist.y, dist.length()))
        dec = output.index(max(output))
        # move up
        if dec == 0:
            self.player.y -= self.speed
        # move down
        elif dec == 1:
            self.player.y += self.speed
        # move left
        elif dec == 2:
            self.player.x -= self.speed
        # move right
        elif dec == 3:
            self.player.x += self.speed
        
        
        if not (self.player.x > 0 and self.player.x < self.width and self.player.y > 0 and self.player.y < self.height):
            self.genome.fitness -= 2
            return

        if self.player.colliderect(self.target):
            self.genome.fitness += 1
            return

        # reward based on new distance to target
        new_dist = (pg.math.Vector2(self.player.center) - pg.math.Vector2(self.target.center)).length()
        new_dist = new_dist if new_dist >= 0 else 1
        if new_dist < dist.length():
            self.genome.fitness += 0.15
        else:
            self.genome.fitness -= 0.1
        
        
        # # reward based on new distance to target
        # new_dist = pg.math.Vector2(self.player.center) - pg.math.Vector2(self.target.center)
        # if new_dist.length() < dist.length():
        #     self.genome.fitness += 10/new_dist.length()
        # else:
        #     self.genome.fitness -= 1


    def test(self, genome, config, follow_mouse=False) -> list:
        global fps
        fps = 100
        self.follow_mouse = follow_mouse
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.record = True
        self.run()
        print(self.genome.fitness)
        return self.record_frames
