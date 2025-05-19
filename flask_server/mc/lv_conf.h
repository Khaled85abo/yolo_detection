#ifndef LV_CONF_H
#define LV_CONF_H

/* Memory settings */
#define LV_MEM_SIZE               (24 * 1024U)
#define LV_MEM_POOL_INCLUDE       <stdlib.h>
#define LV_MEM_POOL_ALLOC         malloc
#define LV_MEM_POOL_FREE          free

/* Graphical settings */
#define LV_LAYER_MAX_MEMORY_USAGE (16 * 1024U)
#define LV_VECTOR_BUFFER_SIZE     (128)

/* Color settings */
#define LV_COLOR_DEPTH            16

/* Font usage */
#define LV_FONT_MONTSERRAT_12     0
#define LV_FONT_MONTSERRAT_14     1
#define LV_FONT_MONTSERRAT_16     0
#define LV_FONT_MONTSERRAT_18     0
#define LV_FONT_MONTSERRAT_20     0
#define LV_FONT_MONTSERRAT_22     0
#define LV_FONT_MONTSERRAT_24     0
#define LV_FONT_MONTSERRAT_26     0
#define LV_FONT_MONTSERRAT_28     0
#define LV_FONT_MONTSERRAT_30     0
#define LV_FONT_MONTSERRAT_32     0
#define LV_FONT_MONTSERRAT_34     0
#define LV_FONT_MONTSERRAT_36     0
#define LV_FONT_MONTSERRAT_38     0
#define LV_FONT_MONTSERRAT_40     0
#define LV_FONT_MONTSERRAT_42     0
#define LV_FONT_MONTSERRAT_44     0
#define LV_FONT_MONTSERRAT_46     0
#define LV_FONT_MONTSERRAT_48     0

/* Feature usage */
#define LV_USE_LOG                0
#define LV_LOG_LEVEL              LV_LOG_LEVEL_WARN
#define LV_USE_ASSERT_NULL        0
#define LV_USE_ASSERT_MALLOC      0
#define LV_USE_ASSERT_MEM_INTEGRITY 0
#define LV_USE_ASSERT_OBJ         0
#define LV_USE_ASSERT_STYLE       0

/* Drawing */
#define LV_USE_DRAW_SW            1
#define LV_USE_DRAW_SW_ASM        0
#define LV_USE_VECTOR_GRAPHIC     0

/* Widget usage */
#define LV_USE_BTN                1
#define LV_USE_DROPDOWN           1
#define LV_USE_LABEL              1
#define LV_USE_OBJ                1
#define LV_USE_TABVIEW            1

/* Disable unnecessary widgets */
#define LV_USE_ANIMIMG            0
#define LV_USE_ARC                0
#define LV_USE_BAR                0
#define LV_USE_BTNMATRIX          0
#define LV_USE_CALENDAR           0
#define LV_USE_CANVAS             0
#define LV_USE_CHART              0
#define LV_USE_CHECKBOX           0
#define LV_USE_COLORWHEEL         0
#define LV_USE_IMGBTN             0
#define LV_USE_IMG                0
#define LV_USE_KEYBOARD           0
#define LV_USE_LED                0
#define LV_USE_LINE               0
#define LV_USE_LIST               0
#define LV_USE_MENU               0
#define LV_USE_METER              0
#define LV_USE_MSGBOX             0
#define LV_USE_ROLLER             0
#define LV_USE_SLIDER             0
#define LV_USE_SPAN               0
#define LV_USE_SPINBOX            0
#define LV_USE_SPINNER            0
#define LV_USE_SWITCH             0
#define LV_USE_TEXTAREA           0
#define LV_USE_TABLE              0
#define LV_USE_TILEVIEW           0
#define LV_USE_WIN                0

/* Themes */
#define LV_USE_THEME_DEFAULT      1
#define LV_USE_THEME_BASIC        0
#define LV_USE_THEME_MONO         0

/* Layouts - MUST ENABLE FLEX for tabview to work */
#define LV_USE_FLEX               1
#define LV_USE_GRID               0

/* Enable needed widgets */
#define LV_WIDGETS_HAS_DEFAULT_VALUE 0

#endif /*LV_CONF_H*/