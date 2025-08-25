# camera.py

import numpy as np
import pyrr
import pygame # Klavye sabitleri (K_w, vb.) için gerekli

# --- HATA DÜZELTMESİ: Importları dosyanın en başına taşıdık ---
from settings import CHUNK_HEIGHT, CHUNK_WIDTH, CHUNK_DEPTH

class Camera:
    def __init__(self):
        # Değişkenler artık burada kullanılabilir olduğu için başlangıç pozisyonunu doğru şekilde ayarlayabiliriz
        self.position = pyrr.Vector3([CHUNK_WIDTH * 1.5, CHUNK_HEIGHT, CHUNK_DEPTH * 1.5])
        self.front = pyrr.Vector3([0.0, 0.0, -1.0])
        self.up = pyrr.Vector3([0.0, 1.0, 0.0])
        self.right = pyrr.Vector3([1.0, 0.0, 0.0])

        self.yaw = -90.0
        self.pitch = 0.0
        self.speed = 15.0
        self.sensitivity = 0.1

        # Vektörleri ilk değerlere göre bir kez güncelleyelim
        self.update_vectors()


    def get_view_matrix(self):
        return pyrr.matrix44.create_look_at(self.position, self.position + self.front, self.up)

    def update_vectors(self):
        front = pyrr.Vector3([0.0, 0.0, 0.0])
        front.x = np.cos(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
        front.y = np.sin(np.radians(self.pitch))
        front.z = np.sin(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
        self.front = pyrr.vector.normalize(front)
        self.right = pyrr.vector.normalize(pyrr.vector3.cross(self.front, pyrr.Vector3([0.0, 1.0, 0.0])))
        self.up = pyrr.vector.normalize(pyrr.vector3.cross(self.right, self.front))

    def process_mouse_movement(self, xoffset, yoffset):
        self.yaw += xoffset * self.sensitivity
        self.pitch -= yoffset * self.sensitivity

        if self.pitch > 89.0: self.pitch = 89.0
        if self.pitch < -89.0: self.pitch = -89.0
        self.update_vectors()

    def process_keyboard_input(self, keys, dt):
        velocity = self.speed * dt
        if keys[pygame.K_w]: self.position += self.front * velocity
        if keys[pygame.K_s]: self.position -= self.front * velocity
        if keys[pygame.K_a]: self.position -= self.right * velocity
        if keys[pygame.K_d]: self.position += self.right * velocity
        if keys[pygame.K_SPACE]: self.position += self.up * velocity
        if keys[pygame.K_LSHIFT]: self.position -= self.up * velocity