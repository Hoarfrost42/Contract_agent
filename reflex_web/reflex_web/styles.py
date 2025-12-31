"""
Reflex 样式系统
Theme: Luminous Light Theme (发光亮色主题)
Vibe: Clean, Airy, Frosted Jade
"""

# ========================================================
# 1. COLOR PALETTE & TYPOGRAPHY
# ========================================================
COLORS = {
    "heading": "#1E293B",      # slate-800 (Soft Black)
    "body": "#64748B",         # slate-500 (Clean Grey)
    "muted": "#94A3B8",        # slate-400 (Muted Grey)
    "accent": "#6366F1",       # indigo-500
    "bg_base": "#F8FAFC",      # slate-50 (Luminous White)
    "white_70": "rgba(255, 255, 255, 0.70)",
    "white_60": "rgba(255, 255, 255, 0.60)",
}

FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Nunito', sans-serif"

# ========================================================
# 2. GLOBAL BACKGROUND (LUMINOUS CANVAS)
# ========================================================
# Base: #F8FAFC
# Orbs: Blue-200 (Top-Left), Purple-200 (Bottom-Right)
LUMINOUS_BG = {
    "background_color": COLORS["bg_base"],
    "background_image": """
        radial-gradient(circle at 15% 15%, rgba(191, 219, 254, 0.4) 0px, transparent 400px), 
        radial-gradient(circle at 85% 85%, rgba(233, 213, 255, 0.4) 0px, transparent 400px)
    """.replace("\n", "").strip(),
    "background_attachment": "fixed",
    "background_size": "cover",
    "min_height": "100vh",
    "position": "fixed",
    "top": "0",
    "left": "0",
    "width": "100vw",
    "height": "100vh",
    "z_index": "-50",
}

# ========================================================
# 3. GLASS MATERIAL (FROSTED JADE)
# ========================================================
# High opacity white, colored shadow, frosted look
GLASS_CARD = {
    "background": COLORS["white_70"],           # bg-white/70
    "backdrop_filter": "blur(32px) saturate(180%)", # backdrop-blur-3xl
    "-webkit-backdrop-filter": "blur(32px) saturate(180%)",
    "border": f"1px solid {COLORS['white_60']}", # border-white/60
    "border_radius": "24px",
    "box_shadow": "0 20px 40px -10px rgba(224, 231, 255, 0.6)", # shadow-xl shadow-indigo-100/60
    "padding": "32px",
}

# Subtle Glass (For secondary containers)
GLASS_CARD_SUBTLE = {
    "background": "rgba(255, 255, 255, 0.5)",
    "backdrop_filter": "blur(20px)",
    "border": "1px solid rgba(255, 255, 255, 0.5)",
    "border_radius": "20px",
    "box_shadow": "0 10px 20px -5px rgba(224, 231, 255, 0.4)",
    "padding": "24px",
}

# Sidebar Glass Style
GLASS_SIDEBAR = {
    "background": "rgba(255, 255, 255, 0.75)",
    "backdrop_filter": "blur(32px) saturate(180%)",
    "border": "1px solid rgba(255, 255, 255, 0.6)",
    "border_radius": "24px",
    "box_shadow": "0 20px 40px -10px rgba(224, 231, 255, 0.5)",
}

# ========================================================
# 4. COMPONENTS
# ========================================================
# Primary Action: Blue-to-Indigo Gradient + Shadow
GRADIENT_BUTTON = {
    "background": "linear-gradient(135deg, #3B82F6 0%, #6366F1 100%)",
    "color": "white",
    "border_radius": "16px",
    "height": "52px",
    "width": "100%",
    "font_weight": "600",
    "font_size": "1.05rem",
    "border": "none",
    "cursor": "pointer",
    "box_shadow": "0 10px 25px -5px rgba(59, 130, 246, 0.4)", # colored shadow
    "_hover": {
        "transform": "translateY(-1px)",
        "box_shadow": "0 15px 30px -5px rgba(59, 130, 246, 0.5)",
    },
    "transition": "all 0.2s ease",
}

GHOST_BUTTON = {
    "background": "rgba(255, 255, 255, 0.6)",
    "color": COLORS["accent"],
    "border_radius": "12px",
    "font_weight": "600",
    "cursor": "pointer",
    "_hover": {
        "background": "rgba(255, 255, 255, 0.9)",
    },
}

UPLOAD_AREA = {
    "border": "2px dashed rgba(99, 102, 241, 0.2)",
    "border_radius": "20px",
    "background": "rgba(255, 255, 255, 0.3)",
    "cursor": "pointer",
    "_hover": {
        "background": "rgba(255, 255, 255, 0.6)",
        "border_color": "rgba(99, 102, 241, 0.4)",
    },
    "transition": "all 0.2s ease",
}

# Pill Tabs
PILL_TAB_CONTAINER = {
    "background": "rgba(255, 255, 255, 0.5)",
    "border_radius": "9999px",
    "padding": "5px",
    "border": "1px solid rgba(255, 255, 255, 0.5)",
}

PILL_TAB_ACTIVE = {
    "background": "white",
    "color": COLORS["heading"],
    "border_radius": "9999px",
    "padding": "10px 24px",
    "font_weight": "600",
    "box_shadow": "0 4px 12px rgba(99, 102, 241, 0.1)",
    "cursor": "pointer",
}

PILL_TAB_INACTIVE = {
    "background": "transparent",
    "color": COLORS["body"],
    "border_radius": "9999px",
    "padding": "10px 24px",
    "font_weight": "500",
    "cursor": "pointer",
    "_hover": {
        "color": COLORS["heading"],
    },
}

# Navigation Items
NAV_ITEM_ACTIVE = {
    "background": "rgba(99, 102, 241, 0.08)",
    "color": COLORS["accent"],
    "font_weight": "600",
    "border_radius": "14px",
}

NAV_ITEM_INACTIVE = {
    "background": "transparent",
    "color": COLORS["body"],
    "font_weight": "500",
    "_hover": {
        "background": "rgba(0,0,0,0.02)",
        "color": COLORS["heading"],
    },
}

# ========================================================
# 5. GLOBAL CSS
# ========================================================
GLOBAL_STYLE = {
    ":root": {
        "--gray-12": f"{COLORS['heading']} !important",
        "--accent-9": f"{COLORS['accent']} !important",
    },
    "body": {
        "background": COLORS["bg_base"],
        "color": COLORS["heading"],
        "font_family": FONT_FAMILY,
    },
    "[role='tab']": {
        "color": f"{COLORS['body']} !important",
    },
}
