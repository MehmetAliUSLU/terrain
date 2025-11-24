# Voxel Terrain Editor

A powerful 3D terrain editing tool built with Python, Pygame, and OpenGL. This application allows users to sculpt terrain, paint textures, and create complex river systems in real-time.

## Features

### Terrain Sculpting
- **Raise/Lower:** Intuitively modify terrain height.
- **Flatten:** Level terrain to a specific height.
- **Smooth:** Soften rough edges and blend terrain features.
- **Slope:** Create smooth slopes between points.
- **Memory Slope:** Copy a specific slope angle and apply it elsewhere.

### Texture Painting
- **Multi-texture Support:** Paint with Grass, Dirt, and Rock textures.
- **Splatmapping:** Smooth blending between different textures based on height and slope.

### River System
- **Path Creation:** Define river paths using control points.
- **Carving:** Automatically carve riverbeds with adjustable width, depth, and smoothing.
- **Visual Feedback:** Real-time preview of the river path before carving.

### Advanced Tools
- **Brush Customization:**
  - Adjustable Size and Strength.
  - Shapes: Circular and Square brushes.
  - Softness control for hard or soft edges.
- **Undo/Redo System:** Full history support for all terrain modifications.
- **Dynamic Preview:** Visual cursor showing the brush influence on the terrain.

### Performance
- **Optimized Meshing:** Uses Numba for fast mesh generation.
- **Threaded Processing:** Background chunk updates to keep the UI responsive.
- **OpenGL Rendering:** Efficient 3D rendering with custom shaders.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/terrain-editor.git
    cd terrain-editor
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main application:
```bash
python main.py
```

## Controls

### Camera
- **Orbit:** Middle Mouse Button
- **Pan:** Ctrl + Middle Mouse Button
- **Zoom:** Mouse Wheel

### Editing
- **Use Tool:** Left Mouse Button
- **River Tool:**
  - **Add Point:** Left Click
  - **Clear Path:** "Yolu Temizle" button in UI
  - **Carve:** "Dere Yatağını Oluştur" button in UI

### Shortcuts
- **Undo:** Ctrl + Z
- **Redo:** Ctrl + Y
- **Quit:** Esc

## Dependencies

- pygame
- PyOpenGL
- numpy
- pyrr
- noise
- numba
- Pillow
- imgui[pygame]

## Project Structure

- `main.py`: Entry point and main application loop.
- `world.py`: Terrain data structure, chunk management, and mesh generation.
- `editor.py`: UI implementation and tool logic.
- `camera.py`: Camera control system.
- `history.py`: Undo/Redo system implementation.
- `settings.py`: Configuration constants and shader code.
- `textures/`: Directory containing texture assets.

## License

[MIT License](LICENSE)
