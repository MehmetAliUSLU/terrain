# main.py

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
import sys
import threading
import queue
import os
from PIL import Image

import imgui
from imgui.integrations.pygame import PygameRenderer

from settings import SCREEN_WIDTH, SCREEN_HEIGHT, VERTEX_SHADER, FRAGMENT_SHADER, CURSOR_VERTEX_SHADER, CURSOR_FRAGMENT_SHADER, CHUNK_WIDTH, CHUNK_DEPTH
from camera import Camera
from world import World
import streaming
from editor import Editor
from history import UndoManager, EditAction

def mesh_worker(request_queue, result_queue):
    from world import create_heightmap_mesh # Numba'nın thread-safe olması için import'u buraya alıyoruz
    while True:
        try:
            chunk = request_queue.get(timeout=2)
            if chunk is None: break
            mesh_data = create_heightmap_mesh(chunk.heightmap)
            result_queue.put((chunk, mesh_data))
        except queue.Empty: continue

class App:
    def __init__(self):
        pygame.init()
        self.screen_size = (SCREEN_WIDTH, SCREEN_HEIGHT)
        self.window = pygame.display.set_mode(self.screen_size, DOUBLEBUF | OPENGL | RESIZABLE)
        
        print("OpenGL Version:", glGetString(GL_VERSION).decode())
        print("GLSL Version:", glGetString(GL_SHADING_LANGUAGE_VERSION).decode())
        print("Vendor:", glGetString(GL_VENDOR).decode())
        print("Renderer:", glGetString(GL_RENDERER).decode())
        
        
        imgui.create_context()
        self.renderer = PygameRenderer()
        self.io = imgui.get_io()
        self.io.display_size = self.screen_size
        pygame.mouse.set_visible(True)
        pygame.event.set_grab(False)
        
        self.shader = self.create_shader_program(VERTEX_SHADER, FRAGMENT_SHADER)
        self.cursor_shader = self.create_shader_program(CURSOR_VERTEX_SHADER, CURSOR_FRAGMENT_SHADER)

        self.undo_manager = UndoManager()
        self.current_action = None # O an devam eden eylemi tutar

        self.texture_ids = {}
        self.texture_units = {}
        self.load_textures()
        self.slope_target_normal = None
        self.slope_initial_pos = None
        
        # Hafızada kalıcı olarak saklanan, kopyalanmış eğim yönü
        self.slope_memory_normal = None
        # Sadece o anki fırça darbesi için geçerli olan başlangıç noktası
        self.stroke_start_pos = None
        
        # YENİ: Dere Yatağı yolu için değişkenler
        self.river_path_points = []
        self.draped_path_vertices = [] # YENİ: Araziye giydirilmiş görsel yol
        self.river_path_vao = None
        self.river_path_vbo = None


        self.camera = Camera()
        self.meshing_request_queue = queue.Queue()
        self.meshing_result_queue = queue.Queue()
        self.world = World(self.meshing_request_queue)
        self.worker_thread = threading.Thread(target=mesh_worker, args=(self.meshing_request_queue, self.meshing_result_queue), daemon=True)
        self.worker_thread.start()
        # Spawn initial chunk ring around camera target
        streaming.update_streaming(self.world, self.camera.target)
        self.editor = Editor()
        self.clock = pygame.time.Clock()
        
        self.flatten_target_height = None
        self.camera_orbiting = False
        self.camera_panning = False
        self.pending_orbit_pivot = None
        
        self.cursor_pos = None
        self.setup_cursor()

        self.dynamic_preview_vao = None
        self.dynamic_preview_vbo = None
        self.dynamic_preview_ebo = None
        self.dynamic_preview_index_count = 0
        self.setup_dynamic_preview_objects()

        # Avoid camera jump on click: start orbit/pan only after tiny drag
        self.drag_threshold_pixels = 3
        self._orbit_drag_accum = 0.0
        self._pan_drag_accum = 0.0
        self._orbit_active = False
        self._pan_active = False


    def generate_draped_path_vertices(self):
        """
        Ana yol noktalarından araziye giydirilmiş, detaylı bir görsel yol oluşturur.
        """
        if len(self.river_path_points) < 2:
            return []

        detailed_points = []
        # Her bir segmenti (iki ana nokta arası) ayrı ayrı işle
        for i in range(len(self.river_path_points) - 1):
            p1 = self.river_path_points[i]
            p2 = self.river_path_points[i+1]

            segment_vector = p2 - p1
            distance = np.linalg.norm(segment_vector)
            if distance == 0:
                continue

            # Mesafe ne kadar uzunsa, o kadar çok ara nokta oluştur (örneğin her birimde bir nokta)
            num_steps = int(distance) + 1

            for step in range(num_steps + 1):
                t = step / num_steps
                # XZ düzleminde ara noktayı hesapla
                interp_point_xz = p1 + t * segment_vector

                # Ara noktanın bulunduğu yerdeki arazi yüksekliğini al
                terrain_y = self.world.get_height_at_world_pos(interp_point_xz.x, interp_point_xz.z)

                # Nihai 3D noktayı oluştur ve Z-fighting olmaması için biraz yükselt
                final_point = pyrr.Vector3([interp_point_xz.x, terrain_y + 0.2, interp_point_xz.z])
                
                # Başlangıç noktası hariç diğerlerini ekle (çakışmayı önlemek için)
                if step > 0 or i == 0:
                    detailed_points.append(final_point)
        
        return detailed_points
    

    def create_shader_program(self, vertex_src, fragment_src):
        return compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))

    def load_textures(self):
        texture_names = ["grass", "dirt", "rock"]
        for i, name in enumerate(texture_names):
            texture_id = glGenTextures(1)
            glActiveTexture(GL_TEXTURE0 + i)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

            try:
                img_path = os.path.join("textures", f"{name}.png")
                img = Image.open(img_path).convert("RGBA")
                img_data = np.array(list(img.getdata()), np.uint8)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
                glGenerateMipmap(GL_TEXTURE_2D)
                
                self.texture_ids[name] = texture_id
                self.texture_units[name] = i
            except FileNotFoundError:
                print(f"UYARI: Doku dosyası bulunamadı: {img_path}")
        
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)

    def setup_cursor(self):
        vertices = np.array([-0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5], dtype=np.float32)
        indices = np.array([0,1,2,2,3,0, 4,5,6,6,7,4, 0,3,7,7,4,0, 1,2,6,6,5,1, 0,1,5,5,4,0, 3,2,6,6,7,3], dtype=np.uint32)
        self.cursor_vao = glGenVertexArrays(1)
        glBindVertexArray(self.cursor_vao)
        vbo = glGenBuffers(1); glBindBuffer(GL_ARRAY_BUFFER, vbo); glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        ebo = glGenBuffers(1); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo); glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0)); glEnableVertexAttribArray(0)
        glBindVertexArray(0)

    def setup_dynamic_preview_objects(self):
        segments = 32
        indices = []
        for i in range(segments):
            indices.extend([0, i + 1, ((i + 1) % segments) + 1])
        
        indices = np.array(indices, dtype=np.uint32)
        self.dynamic_preview_index_count = len(indices)

        self.dynamic_preview_vao = glGenVertexArrays(1)
        glBindVertexArray(self.dynamic_preview_vao)
        
        self.dynamic_preview_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.dynamic_preview_vbo)
        glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)
        
        self.dynamic_preview_ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.dynamic_preview_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

    def setup_river_path_renderer(self):
        """Yol çizgisini renderlamak için gerekli VBO ve VAO'yu hazırlar."""
        self.river_path_vao = glGenVertexArrays(1)
        glBindVertexArray(self.river_path_vao)
        
        self.river_path_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.river_path_vbo)
        # Veri sürekli değişeceği için DYNAMIC_DRAW kullanıyoruz
        glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        
        glBindVertexArray(0)

    def update_river_path_vbo(self, path_data):
        """Yol noktalarını GPU'ya gönderir."""
        if not path_data:
            # VBO'yu temizle
            glBindBuffer(GL_ARRAY_BUFFER, self.river_path_vbo)
            glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            return
        
        # Noktaları GPU'ya göndermek için uygun formata getir
        vertices = np.array(path_data, dtype=np.float32)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.river_path_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def run(self):
        while True:
            dt = self.clock.tick(60) / 1000.0
            if self.handle_events_and_inputs(dt) is False: break
            # Stream chunks around camera target for infinite world
            streaming.update_streaming(self.world, self.camera.target)
            self.process_meshing_results()
            self.render_scene()
        self.quit()

    def process_meshing_results(self):
        try:
            while True:
                chunk, mesh_data = self.meshing_result_queue.get_nowait()
                chunk.upload_mesh_to_gpu(mesh_data)
                chunk.is_meshing = False
                chunk.is_dirty = False
        except queue.Empty: pass

    def handle_events_and_inputs(self, dt):
        # --- 1. Adım: Olay İşleme Döngüsü ---
        # Önce tüm Pygame olaylarını işleyip ImGui'ye bildiriyoruz.
        # Bu, io.key_ctrl gibi durumların güncellenmesini sağlar.
        # Sadece bir kez tetiklenmesi gereken eylemler (tuşa basma anı gibi) burada ele alınır.
        # Not: Aşağıdaki projection_matrix hesaplaması, fare olayları sırasında
        # ray hesaplamalarında kullanılıyor. Önceden yalnızca orta tuş basıldığında
        # oluşturulduğu için diğer tuş olaylarında UnboundLocalError oluşuyordu.
        projection_matrix = pyrr.matrix44.create_perspective_projection_matrix(
            45, self.screen_size[0] / self.screen_size[1], 0.1, 1000
        )
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.meshing_request_queue.put(None)
                return False
            
            self.renderer.process_event(event)
            
            if event.type == VIDEORESIZE: 
                self.screen_size = (event.w, event.h)
                glViewport(0, 0, event.w, event.h)
                self.io.display_size = self.screen_size
                # Ekran boyutu değiştiğinde projeksiyon matrisini güncelle
                projection_matrix = pyrr.matrix44.create_perspective_projection_matrix(
                    45, self.screen_size[0] / self.screen_size[1], 0.1, 1000
                )
            
            # Geri Alma/Yineleme (Ctrl+Z/Y basılma anı)
            if event.type == pygame.KEYDOWN:
                if self.io.key_ctrl and event.key == pygame.K_z:
                    self.undo_manager.undo()
                    for chunk in self.world.chunks.values():
                        if chunk.is_dirty and not chunk.is_meshing:
                            chunk.is_meshing = True
                            self.meshing_request_queue.put(chunk)
                if self.io.key_ctrl and event.key == pygame.K_y:
                    self.undo_manager.redo()
                    for chunk in self.world.chunks.values():
                        if chunk.is_dirty and not chunk.is_meshing:
                            chunk.is_meshing = True
                            self.meshing_request_queue.put(chunk)

            # Fare girdilerini sadece ImGui kullanmıyorsa işle
            if not self.io.want_capture_mouse:
                if event.type == pygame.MOUSEWHEEL:
                    inversion_settings = self.editor.get_inversion_settings()
                    if self.io.key_ctrl:
                        step = event.y
                        if inversion_settings["zoom"]: step *= -1
                        move_amount = step * 0.1 * self.camera.distance * getattr(self.camera, "pan_speed_factor", 1.0)
                        self.camera.target += self.camera.front * move_amount
                        self.camera.update_camera_vectors()
                    else:
                        self.camera.process_zoom(event.y, inversion_settings["zoom"])
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 2:  # Orta fare tuşu
                        if self.io.key_ctrl:
                            self.camera_panning = True
                            self._pan_drag_accum = 0.0
                            self._pan_active = False
                        else:
                            self.camera_orbiting = True
                            self._orbit_drag_accum = 0.0
                            self._orbit_active = False
                        pygame.mouse.get_rel()  # Göreceli hareketi sıfırla
                        pygame.event.set_grab(True)
                        pygame.mouse.set_visible(False)
                        pygame.mouse.get_rel()  # Greceli hareketi sıfırla
                        # Orbit/pan başlatıldığında pivotu imlecin altındaki araziye sabitle (yalnızca ORBIT)
                        if not self.io.key_ctrl:
                                if self.editor.get_camera_options().get("orbit_to_cursor", False):
                                    projection_matrix = pyrr.matrix44.create_perspective_projection_matrix(45, self.screen_size[0] / self.screen_size[1], 0.1, 1000)
                                    ray_params = self.camera.get_ray_from_mouse(pygame.mouse.get_pos(), self.screen_size[0], self.screen_size[1], projection_matrix)
                                    pivot_pos = self.world.raycast_terrain(*ray_params)
                                    if pivot_pos is not None:
                                        # Sadece hareket başladıktan sonra uygulamak için beklet
                                        self.pending_orbit_pivot = pivot_pos
                    ray_params = self.camera.get_ray_from_mouse(pygame.mouse.get_pos(), self.screen_size[0], self.screen_size[1], projection_matrix)
                    cursor_pos = self.world.raycast_terrain(*ray_params)
                    
                    if cursor_pos is not None:
                        selected_tool = self.editor.get_selected_tool()
                        ""
                        # Sol Tık Olayları
                        if event.button == 1:
                            if selected_tool == "river":
                                point_to_add = cursor_pos + pyrr.Vector3([0, 0.2, 0])
                                self.river_path_points.append(point_to_add)
                                # DEĞİŞİKLİK: Detaylı yolu yeniden oluştur ve VBO'yu güncelle
                                self.draped_path_vertices = self.generate_draped_path_vertices()
                                self.update_river_path_vbo(self.draped_path_vertices)
                            elif selected_tool == "memory_slope" and self.io.key_ctrl:
                                self.slope_memory_normal = self.world.get_normal_at_world_pos(cursor_pos.x, cursor_pos.z)
                                print(f"Eğim Kopyalandı! Normal: {self.slope_memory_normal}")
                            else:
                                # Ensure the chunk exists where we start editing
                                chunk = streaming.ensure_chunk_at_world_pos(self.world, cursor_pos.x, cursor_pos.z)
                                if chunk: self.current_action = EditAction()
                                self.stroke_start_pos = cursor_pos
                                if selected_tool == "flatten":
                                    self.flatten_target_height = cursor_pos.y - 0.1

                        # Sağ Tık Olayları
                        if event.button == 3:
                            if selected_tool == "memory_slope" and self.io.key_ctrl:
                                self.slope_memory_normal = None
                                print("Kopyalanan eğim sıfırlandı.")

                if event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 2:  # Orta fare tuşu
                        self.camera_orbiting = False
                        self.camera_panning = False
                        self.pending_orbit_pivot = None
                        self._orbit_active = False
                        self._pan_active = False
                        pygame.event.set_grab(False)
                        pygame.mouse.set_visible(True)
                    if event.button == 1:
                        if self.current_action:
                            self.undo_manager.register_action(self.current_action)
                            self.current_action = None
                        self.flatten_target_height = None
                        self.stroke_start_pos = None

        # --- 2. Adım: Durum Tabanlı İşlemler (Olay döngüsünden sonra) ---
        # Bu çağrı, olay döngüsünden SONRA ve new_frame'den ÖNCE olmalıdır.
        # Klavye ve farenin anlık durumunu ImGui'ye bildirir.
        self.renderer.process_inputs()

        # --- 3. Adım: ImGui Çerçevesini Başlat ---
        # Artık ImGui'nin girdi durumu tutarlı olduğu için yeni bir çerçeve başlatabiliriz.
        imgui.new_frame()
        
        # --- 4. Adım: Arayüzü Çiz ---
        ui_action = self.editor.draw_ui()
        # Kamera hiz ayarlarini Editor'den uygula
        speeds = self.editor.get_control_speeds()
        self.camera.sensitivity = speeds["orbit"]
        self.camera.zoom_sensitivity = speeds["zoom"]
        self.camera.pan_speed_factor = speeds["pan"]

        # --- Hover Info HUD (Top-Right) ---
        # Shows camera target, cursor position and chunk coords
        info_width, info_height = 260, 90
        pos_x = self.screen_size[0] - info_width - 10
        pos_y = 10
        imgui.set_next_window_position(pos_x, pos_y, imgui.ALWAYS)
        imgui.set_next_window_size(info_width, info_height, imgui.ALWAYS)
        flags = imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_TITLE_BAR
        imgui.begin("Konum Bilgisi", flags=flags)

        cam_t = self.camera.target
        cam_chunk_x = int(np.floor(cam_t.x / CHUNK_WIDTH))
        cam_chunk_z = int(np.floor(cam_t.z / CHUNK_DEPTH))
        imgui.text(f"Hedef: x={cam_t.x:.1f} y={cam_t.y:.1f} z={cam_t.z:.1f}")
        imgui.text(f"Hedef Chunk: ({cam_chunk_x}, {cam_chunk_z})")

        if self.cursor_pos is not None:
            cur = self.cursor_pos
            cur_chunk_x = int(np.floor(cur.x / CHUNK_WIDTH))
            cur_chunk_z = int(np.floor(cur.z / CHUNK_DEPTH))
            imgui.text(f"İmleç: x={cur.x:.1f} y={cur.y:.1f} z={cur.z:.1f}")
            imgui.text(f"İmleç Chunk: ({cur_chunk_x}, {cur_chunk_z})")
        else:
            imgui.text("İmleç: -")

        imgui.end()

        # Arayüzden gelen buton eylemlerini işle
        if ui_action == "clear_river_path":
            self.river_path_points.clear()
            # DEĞİŞİKLİK: Detaylı yolu da temizle
            self.draped_path_vertices.clear()
            self.update_river_path_vbo(self.draped_path_vertices)
        elif ui_action == "carve_river":
            if len(self.river_path_points) > 1:
                chunk = self.world.chunks.get((0,0,0)) # Tek chunk varsayımı
                if chunk:
                    action = EditAction()
                    # Dere oyma işlemi ana noktaları kullanır (bu doğru)
                    self.world.carve_river_along_path(self.river_path_points, self.editor.get_river_settings(), action)
                    self.undo_manager.register_action(action)
                    self.river_path_points.clear()
                    # DEĞİŞİKLİK: İşlem bitince detaylı yolu da temizle
                    self.draped_path_vertices.clear()
                    self.update_river_path_vbo(self.draped_path_vertices)

        # --- 5. Adım: Sürekli Devam Eden Eylemler ---
        # Arayüz odağı almadıysa, tuş basılı tutulduğunda devam eden eylemleri gerçekleştir.
        if not self.io.want_capture_mouse:
            mouse_pos = pygame.mouse.get_pos()
            mouse_pressed = pygame.mouse.get_pressed()
            projection_matrix = pyrr.matrix44.create_perspective_projection_matrix(45, self.screen_size[0] / self.screen_size[1], 0.1, 1000)

            if not (self.camera_orbiting or self.camera_panning):
                self.cursor_pos = self.world.raycast_terrain(*self.camera.get_ray_from_mouse(mouse_pos, self.screen_size[0], self.screen_size[1], projection_matrix))

            # Sol fare tuşu basılı tutulurken araziyi değiştir
            if mouse_pressed[0] and self.cursor_pos is not None and self.current_action:
                current_normal_to_use = None
                selected_tool = self.editor.get_selected_tool()

                if selected_tool == "slope" and self.stroke_start_pos is not None:
                    current_normal_to_use = self.world.get_normal_at_world_pos(self.stroke_start_pos.x, self.stroke_start_pos.z)
                elif selected_tool == "memory_slope":
                    current_normal_to_use = self.slope_memory_normal

                self.world.modify_terrain(
                    self.cursor_pos, self.editor.get_brush_settings(), self.editor.brush_strength, 
                    selected_tool, current_action=self.current_action, 
                    paint_index=self.editor.get_paint_texture_index(), target_height=self.flatten_target_height,
                    target_normal=current_normal_to_use, stroke_anchor_pos=self.stroke_start_pos
                )

            # Orta fare tuşu basılı tutulurken kamerayı hareket ettir
            if self.camera_orbiting or self.camera_panning:
                mouse_rel = pygame.mouse.get_rel()
                inversion_settings = self.editor.get_inversion_settings()
                if self.camera_orbiting:
                    # Start orbiting only after a small drag to prevent jump
                    self._orbit_drag_accum += abs(mouse_rel[0]) + abs(mouse_rel[1])
                    if not self._orbit_active:
                        if self._orbit_drag_accum > self.drag_threshold_pixels:
                            if self.pending_orbit_pivot is not None:
                                self.camera.retarget_preserve_position(self.pending_orbit_pivot)
                                self.pending_orbit_pivot = None
                            self._orbit_active = True
                    if self._orbit_active:
                        self.camera.process_orbit(mouse_rel[0], mouse_rel[1], inversion_settings["x"], inversion_settings["y"])
                if self.camera_panning:
                    # Start panning only after a small drag
                    self._pan_drag_accum += abs(mouse_rel[0]) + abs(mouse_rel[1])
                    if not self._pan_active:
                        if self._pan_drag_accum > self.drag_threshold_pixels:
                            self._pan_active = True
                    if self._pan_active:
                        self.camera.process_pan(mouse_rel[0], mouse_rel[1], inversion_settings["x"], inversion_settings["y"])
            
        return True

    def render_scene(self):
        glClearColor(0.5, 0.7, 1.0, 1.0); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        view_matrix = self.camera.get_view_matrix()
        projection_matrix = pyrr.matrix44.create_perspective_projection_matrix(45, self.screen_size[0] / self.screen_size[1], 0.1, 1000)
        
        # --- Ana Arazi Çizimi ---
        glEnable(GL_DEPTH_TEST); glEnable(GL_CULL_FACE); glDisable(GL_BLEND)
        glUseProgram(self.shader)

        for name, unit in self.texture_units.items():
            glActiveTexture(GL_TEXTURE0 + unit)
            glBindTexture(GL_TEXTURE_2D, self.texture_ids[name])

        glUniform1i(glGetUniformLocation(self.shader, "texture_grass"), self.texture_units.get("grass", 0))
        glUniform1i(glGetUniformLocation(self.shader, "texture_dirt"), self.texture_units.get("dirt", 1))
        glUniform1i(glGetUniformLocation(self.shader, "texture_rock"), self.texture_units.get("rock", 2))
        # YENİ: Splatmap doku birimini shader'a bildir
        glUniform1i(glGetUniformLocation(self.shader, "texture_splatmap"), 3) # 3. birimi kullandık

        glUniformMatrix4fv(glGetUniformLocation(self.shader, "projection"), 1, GL_FALSE, projection_matrix)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, view_matrix)
        glUniform3fv(glGetUniformLocation(self.shader, "viewPos"), 1, self.camera.position)
        glUniform3f(glGetUniformLocation(self.shader, "lightPos"), 50, 100, 50)
        glUniform3f(glGetUniformLocation(self.shader, "lightColor"), 1.0, 1.0, 1.0)
        
        self.world.draw(self.shader)

        for unit in self.texture_units.values():
            glActiveTexture(GL_TEXTURE0 + unit)
            glBindTexture(GL_TEXTURE_2D, 0)

        # --- İmleç ve Fırça Önizlemesi Çizimi ---
        if self.cursor_pos is not None:
            glDisable(GL_CULL_FACE); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_FALSE)
            
            glUseProgram(self.cursor_shader)
            
            model_loc = glGetUniformLocation(self.cursor_shader, "model")
            view_loc = glGetUniformLocation(self.cursor_shader, "view")
            proj_loc = glGetUniformLocation(self.cursor_shader, "projection")
            color_loc = glGetUniformLocation(self.cursor_shader, "u_color")

            glUniformMatrix4fv(view_loc, 1, GL_FALSE, view_matrix)
            glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection_matrix)

            # 1. Araziye Uygun Fırça Önizlemesini Oluştur ve Çiz
            glUniformMatrix4fv(model_loc, 1, GL_FALSE, pyrr.matrix44.create_identity())
            glUniform4f(color_loc, 1.0, 1.0, 0.0, 0.5)

            brush_settings = self.editor.get_brush_settings()
            brush_shape = brush_settings["shape"]
            radius = brush_settings["size"] / 2.0
            center_x, center_z = self.cursor_pos.x, self.cursor_pos.z
            
            vertices = []
            indices = []

            if brush_shape == 0: # Dairesel Önizleme
                segments = 32
                # Merkez verteksi ekle
                center_y = self.world.get_height_at_world_pos(center_x, center_z) + 0.05
                vertices.extend([center_x, center_y, center_z])
                # Çember üzerindeki verteksleri ekle
                for i in range(segments):
                    angle = i * (2 * np.pi / segments)
                    world_x = center_x + np.cos(angle) * radius
                    world_z = center_z + np.sin(angle) * radius
                    world_y = self.world.get_height_at_world_pos(world_x, world_z) + 0.05
                    vertices.extend([world_x, world_y, world_z])
                # Üçgen fanı için indeksleri oluştur
                for i in range(segments):
                    indices.extend([0, i + 1, ((i + 1) % segments) + 1])

            elif brush_shape == 1: # Kare Önizleme
                grid_res = 10 # Önizlemenin ne kadar detaylı olacağı
                for i in range(grid_res + 1):
                    for j in range(grid_res + 1):
                        # [-0.5, 0.5] aralığından [center-radius, center+radius] aralığına haritala
                        world_x = center_x + (j / grid_res - 0.5) * brush_settings["size"]
                        world_z = center_z + (i / grid_res - 0.5) * brush_settings["size"]
                        world_y = self.world.get_height_at_world_pos(world_x, world_z) + 0.05
                        vertices.append([world_x, world_y, world_z])
                # Grid için indeksleri oluştur
                for i in range(grid_res):
                    for j in range(grid_res):
                        idx0 = i * (grid_res + 1) + j
                        idx1 = idx0 + 1
                        idx2 = (i + 1) * (grid_res + 1) + j
                        idx3 = idx2 + 1
                        indices.extend([idx0, idx2, idx1])
                        indices.extend([idx1, idx2, idx3])

            vertices = np.array(vertices, dtype=np.float32).flatten()
            indices = np.array(indices, dtype=np.uint32)
            
            glBindVertexArray(self.dynamic_preview_vao)
            
            glBindBuffer(GL_ARRAY_BUFFER, self.dynamic_preview_vbo)
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
            
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.dynamic_preview_ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_DYNAMIC_DRAW)
            
            glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
            
            glBindVertexArray(0)
            glDepthMask(GL_TRUE)      # Derinlik yazmayı tekrar AÇ
            glEnable(GL_CULL_FACE)    # Arka yüz kırpmayı tekrar AÇ
            glDisable(GL_BLEND)       # Saydamlığı KAPAT

        if self.editor.get_selected_tool() == "river" and self.draped_path_vertices:
            glUseProgram(self.cursor_shader) # İmlecin shader'ını kullanabiliriz
            
            # Uniform'ları tekrar ayarla
            glUniformMatrix4fv(glGetUniformLocation(self.cursor_shader, "model"), 1, GL_FALSE, pyrr.matrix44.create_identity())
            glUniformMatrix4fv(glGetUniformLocation(self.cursor_shader, "view"), 1, GL_FALSE, view_matrix)
            glUniformMatrix4fv(glGetUniformLocation(self.cursor_shader, "projection"), 1, GL_FALSE, projection_matrix)
            glUniform4f(glGetUniformLocation(self.cursor_shader, "u_color"), 0.0, 0.5, 1.0, 0.8) # Mavi bir renk

            glLineWidth(3.0) # Çizgi kalınlığı
            glBindVertexArray(self.river_path_vao)
            # DEĞİŞİKLİK: Detaylı yolun uzunluğunu kullan
            glDrawArrays(GL_LINE_STRIP, 0, len(self.draped_path_vertices))
            glBindVertexArray(0)
            glLineWidth(1.0) # Kalınlığı sıfırla



        glUseProgram(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        glDisable(GL_CULL_FACE)
        glDisable(GL_DEPTH_TEST)
        # Aktif doku biriminin de sıfırlandığından emin olalım
        glActiveTexture(GL_TEXTURE0)

        # --- ImGui Arayüz Çizimi ---

        imgui.render()
        self.renderer.render(imgui.get_draw_data())
        
        pygame.display.flip()
        pygame.display.set_caption(f"Voxel Editor - FPS: {self.clock.get_fps():.2f}")
    
    def quit(self):
        self.worker_thread.join()
        self.renderer.shutdown()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = App()
    app.setup_river_path_renderer()
    app.run()


