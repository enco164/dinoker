import numpy as np
import pyautogui
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains


class Environment(object):
    def __init__(self, url="http://wayou.github.io/t-rex-runner/"):
        self.url = url
        self.webdriver = webdriver.Chrome("./chromedriver")
        self.webdriver.get(self.url)
        self.runner_canvas = self.webdriver.find_element_by_class_name("runner-canvas")
        self.webdriver_actions = ActionChains(self.webdriver)
        self.state = np.array([])

    def reset(self):
        self.state = np.array([])
        self.webdriver_actions\
            .move_to_element(self.runner_canvas)\
            .key_down(Keys.SPACE).key_up(Keys.SPACE).perform()
        time.sleep(0.5)
        crashed = self.webdriver.execute_script("return Runner.instance_.crashed")
        return not crashed

    def get_js_state_script(self):
        return \
            "var speed = Runner.instance_.currentSpeed;" \
            "var tRexXPos = Runner.instance_.tRex.xPos;" \
            "var tRexYPos = Runner.instance_.tRex.yPos;" \
            "var obstacle1 = Runner.instance_.horizon.obstacles[0];" \
            "var obstacle2 = Runner.instance_.horizon.obstacles[1];" \
            "var obstacle3 = Runner.instance_.horizon.obstacles[2];" \
            "var obstacle1Width = 0;" \
            "var obstacle1Height = 0;" \
            "var obstacle1XPos = -0.5;" \
            "var obstacle1YPos = -0.5;" \
            "var obstacle2Width = 0;" \
            "var obstacle2Height = 0;" \
            "var obstacle2XPos = -0.5;" \
            "var obstacle2YPos = -0.5;" \
            "var obstacle3Width = 0;" \
            "var obstacle3Height = 0;" \
            "var obstacle3XPos = -0.5;" \
            "var obstacle3YPos = -0.5;" \
            "if (obstacle1) {" \
            "   obstacle1Width = obstacle1.width / 100;" \
            "   obstacle1Height = obstacle1.typeConfig.height / 150;" \
            "   obstacle1XPos = obstacle1.xPos / 600;" \
            "   obstacle1YPos = obstacle1.yPos / 150;" \
            "}" \
            "if (obstacle2) {" \
            "   obstacle2Width = obstacle2.width / 100;" \
            "   obstacle2Height = obstacle2.typeConfig.height / 150;" \
            "   obstacle2XPos = obstacle2.xPos / 600;" \
            "   obstacle2YPos = obstacle2.yPos / 150;" \
            "}"\
            "if (obstacle3) {" \
            "   obstacle3Width = obstacle3.width / 100;" \
            "   obstacle3Height = obstacle3.typeConfig.height / 150;" \
            "   obstacle3XPos = obstacle3.xPos / 600;" \
            "   obstacle3YPos = obstacle3.yPos / 150;" \
            "}" \
            "return [speed / 25., " \
            "tRexXPos / 100, tRexYPos / 150," \
            "obstacle1Width, obstacle1Height, obstacle1XPos, obstacle1YPos," \
            "obstacle2Width, obstacle3Height, obstacle2XPos, obstacle2YPos," \
            "obstacle3Width, obstacle3Height, obstacle3XPos, obstacle3YPos];"
            # speed: 6-20
            # width: 0-36

    def get_state(self):
        states = list()
        states.extend(self.webdriver.execute_script(self.get_js_state_script()))
        return np.array(states)

    def is_game_over(self):
        return self.webdriver.execute_script("return Runner.instance_.crashed")

    def act(self, action):
        if action == 0:
            pyautogui.keyDown('down', pause=0.10)
            pyautogui.keyUp('down')
        elif action == 2:
            pyautogui.keyDown('up', pause=0.10)
            pyautogui.keyUp('up')

        new_state = self.get_state()

        game_over = self.is_game_over()
        reward = 0

        if game_over:
            reward = -1
            time.sleep(1)
        elif len(self.state) > 0 and self.state[5] != -0.5 and self.state[5] < new_state[5]:
            reward = 0.1

        self.state = new_state

        return self.state, reward, game_over
