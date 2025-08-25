# editor.py

import imgui

class Editor:
    def __init__(self):
        self.brush_size = 1
        self.selected_block_id = 1
        
        self.ui_width = 250
        self.ui_height = 160
        self.ui_pos_x = 10
        self.ui_pos_y = 10

    def draw_ui(self):
        """ Her karede ImGui arayüzünü çizen ana fonksiyon """

        # --- DÜZELTİLMİŞ KOD (Eski Sürümler İçin) ---
        # Koşulu, keyword argümanı 'cond' yerine konumsal argüman
        # ve imgui.ONCE sabiti ile belirtiyoruz.
        imgui.set_next_window_position(
            self.ui_pos_x, 
            self.ui_pos_y, 
            imgui.ONCE
        )
        
        imgui.set_next_window_size(
            self.ui_width, 
            self.ui_height, 
            imgui.ONCE
        )
        # ----------------------------------------------

        imgui.begin("Araçlar")

        imgui.text("Fırça Ayarları")
        
        changed, self.brush_size = imgui.slider_int(
            "Boyut", 
            self.brush_size, 
            min_value=1, 
            max_value=10
        )

        imgui.separator()

        imgui.text("Blok Ayarları")
        
        changed, self.selected_block_id = imgui.input_int(
            "Blok ID", 
            self.selected_block_id
        )
        
        if self.selected_block_id < 1:
            self.selected_block_id = 1

        imgui.end()