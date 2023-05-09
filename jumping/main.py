import neat
import pickle
import pygame as pg
from Game import Game

# initialize pygame
pg.init()
width, height = 500, 500

def build_clip(frames):
    import moviepy.video.io.ImageSequenceClip

    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames, fps=120)
    clip.write_videofile('../clips/jumping-best.mp4')

def eval_genomes(genomes, config):
    win = pg.display.set_mode((width, height))
    pg.display.set_caption("Game")

    for i, (_, genome) in enumerate(genomes):
        print(round(i/len(genomes) * 100), end=" ")
        genome.fitness = 0
        Game(win, width, height).train(genome, config)

def run_neat(config):
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-49')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(2))

    winner = p.run(eval_genomes, 100)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)

def test_neat():
    with open("best.pickle", "rb") as f:
        genome = pickle.load(f)
    win = pg.display.set_mode((width, height))
    pg.display.set_caption("Game")
    frames = Game(win, width, height).test(genome, config)
    pg.quit()
    return frames

if __name__ == "__main__":
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         "config-feedforward.txt")
    # run_neat(config)
    frames = test_neat()
    build_clip(frames)
