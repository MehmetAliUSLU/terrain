# settings.py

import numpy as np

# --- Ekran Ayarları ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720

# --- Dünya Ayarları (Yükseklik Haritası için) ---
CHUNK_WIDTH = 64  # X eksenindeki grid çözünürlüğü
CHUNK_DEPTH = 64  # Z eksenindeki grid çözünürlüğü

# --- GLSL Shader Kodları ---
VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoords;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoords = aTexCoords; // Doku koordinatlarını Fragment Shader'a geçir
    
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoords;

uniform sampler2D texture_grass;
uniform sampler2D texture_dirt;
uniform sampler2D texture_rock;
uniform sampler2D texture_splatmap; // YENİ: Karışım haritası

uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 viewPos;

void main()
{
    // Dokuları örnekle
    vec4 grassColor = texture(texture_grass, TexCoords * 0.2);
    vec4 dirtColor = texture(texture_dirt, TexCoords * 0.2);
    vec4 rockColor = texture(texture_rock, TexCoords * 0.2);
    
    // Splatmap'ten karışım ağırlıklarını oku
    // TexCoords'u chunk boyutuna bölerek [0,1] aralığına getiriyoruz
    vec3 blend_weights = texture(texture_splatmap, TexCoords / 64.0).rgb;

    // Ağırlıklara göre dokuları karıştır
    vec3 final_color_rgb = grassColor.rgb * blend_weights.r +
                           dirtColor.rgb * blend_weights.g +
                           rockColor.rgb * blend_weights.b;
    
    vec4 finalColor = vec4(final_color_rgb, 1.0);

    // Işıklandırma
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * lightColor;
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    vec3 result = (ambient + diffuse) * finalColor.rgb;
    FragColor = vec4(result, 1.0);
}
"""

# --- İmleç Shader Kodları ---
CURSOR_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

CURSOR_FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;

void main()
{
    FragColor = vec4(1.0, 0.0, 0.0, 0.5); // Yarı saydam kırmızı renk
}
"""

# --- Streaming & Noise Settings ---
# Number of chunk rings to render around the camera (square radius)
RENDER_DISTANCE = 2
# Unload chunks beyond this radius from the camera
UNLOAD_DISTANCE = 3

# Procedural terrain noise parameters
TERRAIN_SCALE = 100.0
NOISE_OCTAVES = 6
NOISE_PERSISTENCE = 0.5
NOISE_LACUNARITY = 2.0
HEIGHT_SCALE = 6.0
