import pygame
from pygame.locals import *
from OpenGL.GL import *
import imgui
from imgui.integrations.pygame import PygameRenderer
import sys

# Basit ayarlar
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720

def main():
    pygame.init()
    pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), DOUBLEBUF | OPENGL | RESIZABLE)

    # ImGui kurulumu
    imgui.create_context()
    renderer = PygameRenderer()
    io = imgui.get_io()
    io.display_size = (SCREEN_WIDTH, SCREEN_HEIGHT)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            renderer.process_event(event)

        renderer.process_inputs()

        # Ekranı temizle
        glClearColor(0.5, 0.7, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        # --- ImGui Çizimi ---
        # Gerekli OpenGL durumlarını manuel olarak ayarlayalım
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        imgui.new_frame()
        
        # Basit bir test penceresi çiz
        imgui.begin("Test Arayüzü")
        imgui.text("Eğer bu yazıyı görüyorsanız, ImGui çalışıyor!")
        imgui.end()
        
        imgui.render()
        renderer.render(imgui.get_draw_data())
        # --------------------

        pygame.display.flip()

    renderer.shutdown()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()