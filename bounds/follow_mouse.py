import pickle
import pygame as pg
from Game import Game
import neat

# initialize pygame
pg.init()
width, height = 500, 500

def build_clip(frames):
    import moviepy.video.io.ImageSequenceClip

    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames, fps=120)
    clip.write_videofile('../clips/bounds-follow-best.mp4')

def test_neat(config):
    with open("best.pickle", "rb") as f:
        genome = pickle.load(f)
    win = pg.display.set_mode((width, height))
    pg.display.set_caption("Testing genome")
    frames = Game(win, width, height).test(genome, config, follow_mouse=True)
    pg.quit()
    return frames 
    
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        "config-feedforward.txt")
frames = test_neat(config)
build_clip(frames)
