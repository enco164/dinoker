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
        self.ms_per_frame = 16.6

    def reset(self):
        self.ms_per_frame = self.webdriver.execute_script("return Runner.instance_.msPerFrame")
        self.state = np.array([])
        self.webdriver_actions\
            .move_to_element(self.runner_canvas)\
            .key_down(Keys.SPACE).key_up(Keys.SPACE).perform()
        return not self.webdriver.execute_script("return Runner.instance_.crashed")

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
            "var obstacle1XPos = 1000;" \
            "var obstacle1YPos = 210;" \
            "var obstacle2Width = 0;" \
            "var obstacle2Height = 0;" \
            "var obstacle2XPos = 1000;" \
            "var obstacle2YPos = 210;" \
            "var obstacle3Width = 0;" \
            "var obstacle3Height = 0;" \
            "var obstacle3XPos = 1000;" \
            "var obstacle3YPos = 210;" \
            "if (obstacle1) {" \
            "   obstacle1Width = obstacle1.width;" \
            "   obstacle1Height = obstacle1.typeConfig.height;" \
            "   obstacle1XPos = obstacle1.xPos;" \
            "   obstacle1YPos = obstacle1.yPos;" \
            "}" \
            "if (obstacle2) {" \
            "   obstacle2Width = obstacle2.width;" \
            "   obstacle2Height = obstacle2.typeConfig.height;" \
            "   obstacle2XPos = obstacle2.xPos;" \
            "   obstacle2YPos = obstacle2.yPos;" \
            "}"\
            "if (obstacle3) {" \
            "   obstacle3Width = obstacle3.width;" \
            "   obstacle3Height = obstacle3.typeConfig.height;" \
            "   obstacle3XPos = obstacle3.xPos;" \
            "   obstacle3YPos = obstacle3.yPos;" \
            "}" \
            "return [(speed-13)/7., " \
            "(tRexXPos-500)/500., (tRexYPos-105)/105.," \
            "(obstacle1Width-35)/35., (obstacle1Height-35)/35., (obstacle1XPos-500)/500., (obstacle1YPos-105)/105.," \
            "(obstacle2Width-35)/35., (obstacle3Height-35)/35., (obstacle2XPos-500)/500., (obstacle2YPos-105)/105.," \
            "(obstacle3Width-35)/35., (obstacle3Height-35)/35., (obstacle3XPos-500)/500., (obstacle3YPos-105)/105.];"
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
            pyautogui.keyDown('down', pause=0.15)
            pyautogui.keyUp('down')
        elif action == 2:
            pyautogui.keyDown('up', pause=0.15)
            pyautogui.keyUp('up')

        game_over = self.is_game_over()
        new_state = self.get_state()

        reward = 0
        if game_over:
            reward = -20
        elif len(self.state) > 0 and self.state[5] != 10000 and self.state[5] < new_state[5]:
            print "===========jumped========="
            reward = 5

        self.state = new_state

        return self.state, reward, self.is_game_over()

    def getUrl(self):
        return self.url