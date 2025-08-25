# main.py

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
import sys

import imgui
from imgui.integrations.pygame import PygameRenderer

from settings import SCREEN_WIDTH, SCREEN_HEIGHT, VERTEX_SHADER, FRAGMENT_SHADER
from camera import Camera
from world import World
from editor import Editor

class App:
    def __init__(self):
        pygame.init()
        self.window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), DOUBLEBUF | OPENGL | RESIZABLE)

        imgui.create_context()
        self.renderer = PygameRenderer()
        self.io = imgui.get_io()
        self.io.display_size = (SCREEN_WIDTH, SCREEN_HEIGHT)

        pygame.mouse.set_visible(True)
        pygame.event.set_grab(False)

        self.shader = self.create_shader_program()
        self.camera = Camera()
        self.world = World()
        self.editor = Editor()
        self.clock = pygame.time.Clock()

        self.setup_uniforms()

    def create_shader_program(self):
        return compileProgram(compileShader(VERTEX_SHADER, GL_VERTEX_SHADER), compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER))

    def setup_uniforms(self):
        glUseProgram(self.shader)
        projection = pyrr.matrix44.create_perspective_projection_matrix(45, SCREEN_WIDTH / SCREEN_HEIGHT, 0.1, 500)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "projection"), 1, GL_FALSE, projection)
        glUniform3f(glGetUniformLocation(self.shader, "lightPos"), 50, 100, 50)
        glUniform3f(glGetUniformLocation(self.shader, "lightColor"), 1.0, 1.0, 1.0)
        glUseProgram(0)

    def run(self):
        while True:
            dt = self.clock.tick(60) / 1000.0
            
            if self.handle_events_and_inputs(dt) is False:
                break
            
            self.render_scene()

        self.quit()

    def handle_events_and_inputs(self, dt):
        self.renderer.process_inputs()
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False
            
            self.renderer.process_event(event)

            if event.type == VIDEORESIZE:
                glViewport(0, 0, event.w, event.h)
                self.io.display_size = (event.w, event.h)

            if not self.io.want_capture_mouse:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    hit_pos, prev_pos = self.world.raycast(self.camera.position, self.camera.front)
                    if event.button == 1 and hit_pos:
                        self.world.modify_voxels_in_radius(np.array(hit_pos), self.editor.brush_size, 0)
                    if event.button == 3 and prev_pos:
                        self.world.modify_voxels_in_radius(np.array(prev_pos), self.editor.brush_size, self.editor.selected_block_id)

        if not self.io.want_capture_mouse:
            keys = pygame.key.get_pressed()
            self.camera.process_keyboard_input(keys, dt)
            if pygame.mouse.get_pressed()[2]:
                if not pygame.event.get_grab(): pygame.event.set_grab(True)
                mouse_rel = pygame.mouse.get_rel()
                self.camera.process_mouse_movement(mouse_rel[0], mouse_rel[1])
            else:
                if pygame.event.get_grab(): pygame.event.set_grab(False)
        return True

    def render_scene(self):
        glClearColor(0.5, 0.7, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # --- AŞAMA 1: 3D DÜNYA ÇİZİMİ ---
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glDisable(GL_BLEND)

        glUseProgram(self.shader)
        
        view_matrix = self.camera.get_view_matrix()
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, view_matrix)
        glUniform3fv(glGetUniformLocation(self.shader, "viewPos"), 1, self.camera.position)
        
        self.world.draw(self.shader)

        # --- AŞAMA 2: OPENGL DURUMUNU TAMAMEN TEMİZLEME ---
        # <<< KRITIK DÜZELTME 1: Shader programını serbest bırak >>>
        glUseProgram(0)
        # <<< KRITIK DÜZELTME 2: Aktif buffer'ları serbest bırak >>>
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        # VAO zaten world.py içinde serbest bırakılıyor (glBindVertexArray(0))

        # --- AŞAMA 3: 2D ARAYÜZ (IMGUI) ÇİZİMİ ---
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        imgui.new_frame()
        self.editor.draw_ui()
        imgui.render()
        self.renderer.render(imgui.get_draw_data())

        # --- AŞAMA 4: EKRANI GÜNCELLE ---
        pygame.display.flip()
        pygame.display.set_caption(f"Voxel Editor - FPS: {self.clock.get_fps():.2f}")
    
    def quit(self):
        self.renderer.shutdown()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = App()
    app.run()