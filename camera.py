# camera.py

import numpy as np
import pyrr
import pygame

from settings import CHUNK_WIDTH, CHUNK_DEPTH

class Camera:
    def __init__(self):
        # Yörünge kamerası için temel değişkenler
        self.target = pyrr.Vector3([CHUNK_WIDTH / 2.0, 0.0, CHUNK_DEPTH / 2.0])
        self.distance = 40.0  # Hedefe olan uzaklık (zoom seviyesi)
        self.yaw = -90.0  # Yanal açı (Azimuth)
        self.pitch = 30.0   # Dikey açı (Elevation)

        self.sensitivity = 0.4
        self.zoom_sensitivity = 2.0

        # Bu değişkenler her karede yeniden hesaplanacak
        self.position = pyrr.Vector3()
        self.up = pyrr.Vector3([0.0, 1.0, 0.0])
        self.right = pyrr.Vector3()
        self.front = pyrr.Vector3() # Artık doğrudan hedefe bakıyor

        self.update_camera_vectors()

    def retarget_preserve_position(self, new_target: pyrr.Vector3):
        """
        Change orbit target to new_target but keep the current camera position
        by recalculating yaw, pitch, and distance from the new target.
        """
        # Vector from new target to current position
        offset = self.position - new_target
        dx, dy, dz = float(offset.x), float(offset.y), float(offset.z)
        dist = float(np.linalg.norm([dx, dy, dz]))
        if dist < 1e-6:
            # Avoid degenerate case; keep a tiny distance in front
            dist = max(0.001, self.distance)
            dx = dist; dy = 0.0; dz = 0.0

        # Compute spherical angles to match current position around new target
        rad_pitch = float(np.arcsin(np.clip(dy / dist, -1.0, 1.0)))
        rad_yaw = float(np.arctan2(dz, dx))

        self.target = pyrr.Vector3([new_target.x, new_target.y, new_target.z])
        self.distance = dist
        self.pitch = np.degrees(rad_pitch)
        self.yaw = np.degrees(rad_yaw)
        self.update_camera_vectors()

    def get_view_matrix(self):
        # Her zaman hedefe bak
        return pyrr.matrix44.create_look_at(self.position, self.target, self.up)

    def update_camera_vectors(self):
        """
        Hedef, uzaklık ve açılara göre kameranın 3D pozisyonunu ve
        yön vektörlerini (front, right, up) hesaplar.
        """
        # Küresel koordinatları Kartezyen koordinatlara çevirerek pozisyonu bul
        # Açıları radyana çevirmeyi unutma!
        rad_pitch = np.radians(self.pitch)
        rad_yaw = np.radians(self.yaw)
        
        x = self.target.x + self.distance * np.cos(rad_pitch) * np.cos(rad_yaw)
        y = self.target.y + self.distance * np.sin(rad_pitch)
        z = self.target.z + self.distance * np.cos(rad_pitch) * np.sin(rad_yaw)
        self.position = pyrr.Vector3([x, y, z])

        # Yeni yön vektörlerini hesapla
        self.front = pyrr.vector.normalize(self.target - self.position)
        self.right = pyrr.vector.normalize(pyrr.vector3.cross(self.front, pyrr.Vector3([0.0, 1.0, 0.0])))
        self.up = pyrr.vector.normalize(pyrr.vector3.cross(self.right, self.front))

    def process_orbit(self, dx, dy, invert_x=False, invert_y=False):
        """ Farenin orta tuşuyla yörüngede dönme işlemini yönetir. """
        # Ayara göre eksenleri tersine çevir
        if invert_x: dx *= -1
        if invert_y: dy *= -1

        self.yaw -= dx * self.sensitivity
        self.pitch -= dy * self.sensitivity

        if self.pitch > 89.0: self.pitch = 89.0
        if self.pitch < 5.0: self.pitch = 5.0 # Yerin altına girmemesi için alt limiti artıralım
        
        self.update_camera_vectors()

    def process_pan(self, dx, dy, invert_x=False, invert_y=False):
        """ Ctrl + Orta tuş ile kaydırma işlemini yönetir. """
        # Ayara göre eksenleri tersine çevir
        if invert_x: dx *= -1
        if invert_y: dy *= -1
        
        pan_speed = 0.002 * self.distance
        if hasattr(self, "pan_speed_factor"):
            pan_speed *= self.pan_speed_factor
        self.target -= self.right * dx * pan_speed
        self.target += self.up * dy * pan_speed # Pan Y yönü genellikle ters hissettirir, bu yüzden + yapıyoruz
        
        self.update_camera_vectors()

    def process_zoom(self, scroll_y, invert=False):
        """ Fare tekerleği ile yakınlaşma/uzaklaşma işlemini yönetir. """
        # Ayara göre yönü tersine çevir
        if invert: scroll_y *= -1
        
        self.distance -= scroll_y * self.zoom_sensitivity

        if self.distance < 2.0: self.distance = 2.0
        if self.distance > 150.0: self.distance = 150.0

        self.update_camera_vectors()

    def get_ray_from_mouse(self, mouse_pos, screen_width, screen_height, projection_matrix):
        """
        Bu fonksiyon artık yeni kamera mantığıyla da doğru çalışır.
        Değişiklik yapmaya gerek yok.
        """
        x = (2.0 * mouse_pos[0]) / screen_width - 1.0
        y = 1.0 - (2.0 * mouse_pos[1]) / screen_height
        ray_clip = np.array([x, y, -1.0, 1.0], dtype=np.float32)

        try:
            inv_projection_transposed = np.linalg.inv(np.array(projection_matrix, dtype=np.float32))
            inv_projection = inv_projection_transposed.T
        except np.linalg.LinAlgError: return self.position, self.front

        ray_eye_untransformed = np.dot(inv_projection, ray_clip)
        ray_eye = np.array([ray_eye_untransformed[0], ray_eye_untransformed[1], -1.0, 0.0], dtype=np.float32)

        try:
            inv_view_transposed = np.linalg.inv(np.array(self.get_view_matrix(), dtype=np.float32))
            inv_view = inv_view_transposed.T
        except np.linalg.LinAlgError: return self.position, self.front
        
        ray_world_4d = np.dot(inv_view, ray_eye)
        ray_world = np.array([ray_world_4d[0], ray_world_4d[1], ray_world_4d[2]], dtype=np.float32)
        
        norm = np.linalg.norm(ray_world)
        if norm > 0.0: ray_world = ray_world / norm
        else: return self.position, self.front

        return self.position, pyrr.Vector3(ray_world)
