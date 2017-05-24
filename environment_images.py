import numpy as np
import skimage
from skimage import transform, color, exposure
import pyautogui
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

from PIL import Image
from mss import mss
import thread


# called by each thread
def do_action(action):
    if action == 0:
        pyautogui.keyDown('down')
        time.sleep(0.1)
        pyautogui.keyUp('down')
    elif action == 2:
        pyautogui.keyDown('up')
        time.sleep(0.1)
        pyautogui.keyUp('up')
    pass


class EnvironmentImages(object):
    def __init__(self, url="http://wayou.github.io/t-rex-runner/"):
        self.url = url
        self.webdriver = webdriver.Chrome("./chromedriver")
        self.webdriver.get(self.url)
        self.runner_canvas = self.webdriver.find_element_by_class_name("runner-canvas")
        self.webdriver_actions = ActionChains(self.webdriver)
        self.webdriver.set_window_size(600, 450)
        self.webdriver.set_window_position(50, 50)
        self.state = np.array([])
        self.ms_per_frame = 16.6
        self.obstacle_pos = None

    def reset(self):
        self.ms_per_frame = self.webdriver.execute_script("return Runner.instance_.msPerFrame")
        self.state = np.array([])
        self.webdriver_actions\
            .move_to_element(self.runner_canvas)\
            .key_down(Keys.SPACE).key_up(Keys.SPACE).perform()
        return not self.webdriver.execute_script("return Runner.instance_.crashed")

    def get_screen(self):
        screen = list()

        with mss() as sct:
            # The screen part to capture
            mon = {'top': 200, 'left': 50, 'width': 600, 'height': 300}
            sct.get_pixels(mon)
            img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
            img = img.convert('L')
            img = np.asarray(img.getdata(), dtype=np.float64).reshape((img.size[1], img.size[0]))
            img = skimage.transform.resize(img, (84, 84))

        return img

    def get_obstacle_pos(self):
        script = \
            "var obstacle1 = Runner.instance_.horizon.obstacles[0];" \
            "var obstacle1XPos = 1000;" \
            "if (obstacle1) {" \
            "   obstacle1XPos = obstacle1.xPos;" \
            "}" \
            "return obstacle1XPos;"
        return self.webdriver.execute_script(script)

    def get_state(self):
        self.obstacle_pos = self.get_obstacle_pos()

        states = list()

        # get 2 screens
        states.extend([self.get_screen()])
        time.sleep(0.016)
        states.extend([self.get_screen()])

        states = np.array(states)
        states = states.reshape(1, 84, 84, 2)

        return states

    def is_game_over(self):
        return self.webdriver.execute_script("return Runner.instance_.crashed")

    def act(self, action):
        if action == 0:
            pyautogui.keyDown('down', pause=0.100)
            pyautogui.keyUp('down')
        elif action == 2:
            pyautogui.keyDown('up', pause=0.100)
            pyautogui.keyUp('up')

        old_obstacle_pos = self.obstacle_pos
        state = self.get_state()

        game_over = self.is_game_over()
        reward = 0
        if game_over:
            reward = -10
        elif old_obstacle_pos != 10000 and old_obstacle_pos < self.obstacle_pos:
            print "===========jumped========="
            reward = 1

        return state, reward, self.is_game_over()
