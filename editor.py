# editor.py

import imgui

class Editor:
    def __init__(self):
        self.brush_size = 3
        self.brush_strength = 0.5
        # YENİ DURUM EKLE: 3: Düzgünleştir
        self.tool_mode = 0  # 0-Yükselt, 1-Alçalt, 2-Düzleştir, 3-Düzgünleştir, 4-Boya, 5-Eğim, 6-Hafızalı Eğim
        self.paint_texture_index = 0 # 0:Çimen, 1:Toprak, 2:Kaya
        
        # YENİ: Dere Yatağı aracı modu (7)
        self.tool_mode = 0  # ..., 6-Hafızalı Eğim, 7-Dere Yatağı
        self.brush_shape = 0  # YENİ: 0 -> Dairesel, 1 -> Kare
        self.brush_softness = 0.5  # YENİ: 0.0 -> En Sert, 1.0 -> En Yumuşak

        # YENİ: Dere yatağı aracı için özel ayarlar
        self.river_width = 4.0 # Dere yatağının genişliği
        self.river_depth = 1.5 # Yatağın ne kadar derin olacağı
        self.river_smoothing = 0.8 # Kenarların ne kadar yumuşatılacağı
        
        self.invert_pan_orbit_x = False
        self.invert_pan_orbit_y = False
        self.invert_zoom = False
        
        self.ui_width = 250
        self.ui_height = 370 # Yüksekliği biraz artıralım
        self.ui_pos_x = 10
        self.ui_pos_y = 10

    def draw_ui(self):
        action_taken = None 
        imgui.set_next_window_position(self.ui_pos_x, self.ui_pos_y, imgui.ONCE)
        imgui.set_next_window_size(self.ui_width, self.ui_height, imgui.ONCE)
        
        imgui.begin("Araçlar")

        imgui.text("Fırça Ayarları")
        _, self.brush_size = imgui.slider_int("Boyut", self.brush_size, 1, 20)
        _, self.brush_strength = imgui.slider_float("Güç", self.brush_strength, 0.01, 1.0)
        _, self.brush_softness = imgui.slider_float("Yumuşaklık", self.brush_softness, 0.0, 1.0)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Fırçanın kenar sertliğini ayarlar.\n0.0: Sert Kenar, 1.0: Yumuşak Kenar")
        
        imgui.separator()
        imgui.text("Araç")
        
        if imgui.radio_button("Yükselt", self.tool_mode == 0): self.tool_mode = 0
        imgui.same_line();
        if imgui.radio_button("Alçalt", self.tool_mode == 1): self.tool_mode = 1
        
        if imgui.radio_button("Düzleştir", self.tool_mode == 2): self.tool_mode = 2
        imgui.same_line();
        if imgui.radio_button("Düzgünleştir", self.tool_mode == 3): self.tool_mode = 3

        if imgui.radio_button("Boya", self.tool_mode == 4): self.tool_mode = 4
        
        if imgui.radio_button("Eğim", self.tool_mode == 5): self.tool_mode = 5
        
        if imgui.radio_button("Hafızalı Eğim", self.tool_mode == 6): self.tool_mode = 6
        
        if imgui.radio_button("Dere Yatağı", self.tool_mode == 7): self.tool_mode = 7
        
        
        imgui.text("Şekil")
        if imgui.radio_button("Dairesel", self.brush_shape == 0): self.brush_shape = 0
        imgui.same_line()
        if imgui.radio_button("Kare", self.brush_shape == 1): self.brush_shape = 1

        # Eğer "Boya" aracı seçiliyse, doku seçeneklerini göster
        if self.tool_mode == 4:
            imgui.separator()
            imgui.text("Boya Dokusu")
            if imgui.radio_button("Çimen", self.paint_texture_index == 0): self.paint_texture_index = 0
            imgui.same_line();
            if imgui.radio_button("Toprak", self.paint_texture_index == 1): self.paint_texture_index = 1
            imgui.same_line();
            if imgui.radio_button("Kaya", self.paint_texture_index == 2): self.paint_texture_index = 2
        
        if self.tool_mode == 6:
            imgui.separator()
            imgui.text_wrapped("Ctrl+Sol Tık: Eğim Kopyala")
            imgui.text_wrapped("Ctrl+Sağ Tık: Eğim Sıfırla")
            
        if self.tool_mode == 7:
            imgui.separator()
            imgui.text("Dere Yatağı Ayarları")
            _, self.river_width = imgui.slider_float("Genişlik", self.river_width, 1.0, 15.0)
            _, self.river_depth = imgui.slider_float("Derinlik", self.river_depth, 0.5, 10.0)
            _, self.river_smoothing = imgui.slider_float("Yumuşatma", self.river_smoothing, 0.1, 1.0)
            
            imgui.separator()
            # 2. Butona basıldığında değişkenin değeri ayarlanır (bu kısım zaten doğruydu)
            if imgui.button("Yolu Temizle"):
                action_taken = "clear_river_path"
            imgui.same_line()
            if imgui.button("Dere Yatağını Oluştur"):
                action_taken = "carve_river"
                
                
        
        imgui.separator()
        imgui.text("Kontrol Ayarları")

        # imgui.checkbox fonksiyonu (değişti_mi, yeni_değer) döndürür
        _, self.invert_pan_orbit_x = imgui.checkbox("Yatay Ekseni Çevir", self.invert_pan_orbit_x)
        _, self.invert_pan_orbit_y = imgui.checkbox("Dikey Ekseni Çevir", self.invert_pan_orbit_y)
        _, self.invert_zoom = imgui.checkbox("Yakınlaştırmayı Çevir", self.invert_zoom)
        
        imgui.end()

        return action_taken

    def get_selected_tool(self):
        if self.tool_mode == 0: return "raise"
        if self.tool_mode == 1: return "lower"
        if self.tool_mode == 2: return "flatten"
        if self.tool_mode == 3: return "smooth"
        if self.tool_mode == 4: return "paint"
        if self.tool_mode == 5: return "slope"
        if self.tool_mode == 6: return "memory_slope"
        if self.tool_mode == 7: return "river"
        return "raise"
        
    def get_brush_settings(self):
        return {
            "size": self.brush_size,
            "strength": self.brush_strength,
            "shape": self.brush_shape, # 0: Daire, 1: Kare
            "softness": self.brush_softness
        }
        
    def get_river_settings(self):
        return {
            "width": self.river_width,
            "depth": self.river_depth,
            "smoothing": self.river_smoothing
        }
    
    def get_paint_texture_index(self):
        return self.paint_texture_index
       
    def get_inversion_settings(self):
        return {
            "x": self.invert_pan_orbit_x,
            "y": self.invert_pan_orbit_y,
            "zoom": self.invert_zoom
        }