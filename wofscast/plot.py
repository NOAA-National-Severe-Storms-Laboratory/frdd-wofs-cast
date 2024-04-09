import numpy as np

class WoFSLevels:
    """
    WoFSLevels contains pre-defined contour levels for the 
    various variables.

    Authors: Montgomery Flora (monte-flora) 
    Email : monte.flora@noaa.gov
    
    """
    pmm_dz_levels           = [35., 50.]
    frp_levels          = [50., 200., 500., 1000.]
    
    ############### Contour Levels:  ##################

    hail_ml_levels          = np.arange(0.25,3.25,0.25)
    uh_2to5_min_levels      = np.arange(-120, 15., 15.) 
    uh_0to2_min_levels      = np.arange(-40, 5., 5.) 
    #min_uh_levels           = np.arange(-40, 0., 5.)
    cape_levels             = [25., 50., 100., 200., 300., 500., 750., 1000., 1500., 2000., 2500., 3000., 4000., 5000., 7000.] #(J Kg^-1)
    cape_levels_legacy      = np.arange(250.,4000.,250.) #(J Kg^-1)
    cin_levels              = np.arange(-200.,25.,25.) #(J Kg^-1)
    temp_levels             = np.arange(50., 110., 5.)              #(deg F)
    #temp_levels_850         = np.arange(25., 85., 5.)               #(deg F)
    #temp_levels_700         = np.arange(10., 70., 5.)               #(deg F)
    the_levels              = np.arange(273., 360., 3.)             #(K)
    td_levels_tc            = np.arange(12., 80., 4.)               #(deg F) 
    td_levels               = np.arange(32., 80., 4.)               #(deg F) 
    #td_levels_700           = np.arange(12., 60., 4.)               #(deg F)
    td_levels_mid           = np.arange(-18,18,3)
    temp_levels_mid         = np.arange(-6, 30,3)

    ws_levels_low           = np.arange(10.,70.,6.) #(kts)
    ws_levels_high          = np.arange(30.,90.,6.) #(kts)
    ws_levels_500           = np.arange(10., 90., 8.)               #(kts)
    ws_levels_low_ms        = np.arange(5.,35.,3.) #(m s^-1)
    ws_levels_high_ms       = np.arange(15.,45.,3.) #(m s^-1)
    srh_levels              = [25., 50., 75., 100., 150., 200., 250., 300., 350., 400., 500., 600., 700., 850., 1000.]  #(m^2 s^-2)
    srh_levels_legacy       = np.arange(40.,640.,40.)  #(m^2 s^-2)
    stp_levels              = np.arange(0.25,7.75,0.5) #(unitless)
    stp_srh0to500_levels    = np.arange(0.25,7.75,0.5) #(unitless)
    swdown_levels           = np.arange(0.,1300.,100.)              #(W m^-2)
    dz_levels_nws           = np.arange(20.0,80.,5.)                #(dBZ)
    pbl_levels              = np.arange(0.,2400.,200.)              #(m)
    mfc_levels              = np.arange(0.,80.,10.)                 #(g/kg*s)
    corf_levels             = np.arange(10.,60.,5.)                 #(m s^-1)
    ul_dvg_levels           = np.arange(-6.,7.,1.)                  #(10^-5 s^-1)
    cp_levels               = np.arange(50,1050,50)                 #(hPa)
    pw_levels               = np.arange(0.0,3.1,0.1)                #(in.)
    #slp_levels_tc          = np.arange(940.,1016.,4)               #(hPa)
    #mslp_levels             = np.arange(970.,1027.,3)               #(hPa)
    
    mslp_levels             = np.arange(1028, 990, -2)
    mslp_levels = mslp_levels[::-1] 
    
    omega_levels            = np.arange(-9.,11.,2.)
    uv_levels               = np.arange(-30.,36.,6.)                #(kts)
    echo_levels             = np.arange(5.,65.,5.)
    
    torn_prob_levels        = np.arange(0,0.80,0.05)
    ml_prob_levels          = np.arange(0,1.1,0.1)
    
    chg_levels              = np.arange(1, 150, 10) 
    scp_levels              = [0.25, 0.5, 1., 2., 4., 6., 8., 10., 15., 20., 25., 30., 40.]  
    
    cwp_levels              = [0.05,0.075,0.1,0.25,0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,4.0,5.0,6.0,8.0,10.0,15.0,30.0,60.]
    rh_levels               = [5., 7.5, 10., 12.5, 15., 17.5, 20., 22.5, 25., 27.5, 30., 32.5, 35., 37.5, 40., 50., 60., 70., 80., 90.]
    fosberg_levels          = [30., 35., 40., 45., 50., 55., 60., 65., 70., 75., 80., 85., 90., 95., 100.]
    rfti_levels             = np.arange(0.5,10.5,1)
    smoke_levels            = [5.,10.,15.,20.,25.,30.,40.,50.,75.,100.,175.,250.,500.,1000.]
    aot_levels              = np.arange(0.1,1.1, 0.1)
#    smokeh_levels           = np.arange(0.5, 10., 0.5) #5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90
    smokeh_levels           = [0.5, 1., 1.5, 2., 2.5, 3., 4., 5., 6., 7., 8., 10.]
    prob_levels             = np.arange(0.1,1.1,0.1)                #(%)
    
    # 1.5 km 
    wz_0to2_levels_1500m           = np.arange(0.004,0.0235,0.0015)      #(s^-1)
    uh_2to5_levels_1500m           = np.arange(60.,840.,60.)               #(m^2 s^-2)
    uh_0to2_levels_1500m           = np.arange(30.,420.,30.)               #(m^2 s^-2)
    w_up_levels_1500m              = np.arange(5.,70.,5.)                  #(m s^-1)

    wz_0to2_levels_3000m           = np.arange(0.002,0.01175,0.00075)      #(s^-1)
    uh_2to5_levels_3000m           = np.arange(25.,350.,25.)               #(m^2 s^-2)
    uh_0to2_levels_3000m           = np.arange(15.,210.,15.)               #(m^2 s^-2)
    w_up_levels_3000m              = np.arange(3.,42.,3.)                  #(m s^-1)
    
    wz_0to2_levels    = np.arange(0.002,0.01175,0.00075)      #(s^-1)
    uh_2to5_levels    = np.arange(25.,350.,25.)               #(m^2 s^-2)
    uh_0to2_levels    = np.arange(10.,140.,10.)               #(m^2 s^-2)
    w_up_levels       = np.arange(3.,42.,3.)                  #(m s^-1)
    
    ws_levels_svr           = np.arange(15.,105.,5.)
    ws_levels_low           = np.arange(10.,70.,6.)                 #(kts)
    ws_levels_high          = np.arange(30.,90.,6.)                 #(kts)
    dz_levels_nws           = np.arange(20.0,80.,5.)                #(dBZ)

    hail_levels             = np.arange(0.5,3.75,0.25)              #(in)
    rain_levels             = [0.01, 0.1, 0.25, 0.50, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]   #(in)
    
    rain_rate_levels        = [0.01,0.03,0.05,0.10, 0.15, 0.20,0.25,0.30, 0.40,0.50,0.60, 0.75, 1.00]
    
    fed_levels              = [1., 2., 4., 8., 12., 16., 20., 25., 35., 50., 75., 100., 150.]  #(flashes 5-min^-1)

    ws_levels_kts_tc        = np.arange(22.,154.,6.)                  #(kts)
    ws_levels_kts_trop      = [34.,64.,96.,137.]                    # TS, H1, H3, H5
    ws_levels_ms_trop       = [18.,33.,50.,70.]

    
    # Satellite Levels ================================================
    ir_levels               = np.arange(188,314,2)          # K
    irbw_levels             = np.arange(188,314,4)          # K
    wv_levels               = np.arange(190,274,2)          # K
    vis_levels              = np.arange(0.025,1.05,0.025)
    


    
    
import matplotlib

class WoFSColors:
    """
    WoFSPlotParameters contains the specific colors/colormaps used for the WoFS webviewer. 
    These colors are single color colortable RGB values 
    from colorbrewer 2.0 (http://colorbrewer2.org/).
    
    Authors: Montgomery Flora, Patrick Skinner,
    
    """
    orange1 = (255 / 255., 245 / 255., 235 / 255.)
    orange2 = (254 / 255., 230 / 255., 206 / 255.)
    orange3 = (253 / 255., 208 / 255., 162 / 255.)
    orange4 = (253 / 255., 174 / 255., 107 / 255.)
    orange5 = (253 / 255., 141 / 255., 60 / 255.)
    orange6 = (241 / 255., 105 / 255., 19 / 255.)
    orange7 = (217 / 255., 72 / 255., 1 / 255.)
    orange8 = (166 / 255., 54 / 255., 3 / 255.)
    orange9 = (127 / 255., 39 / 255., 4 / 255.)

    blue1 = (247/255., 251/255., 255/255.)
    blue2 = (222/255., 235/255., 247/255.)
    blue3 = (198/255., 219/255., 239/255.)
    blue4 = (158/255., 202/255., 225/255.)
    blue5 = (107/255., 174/255., 214/255.)
    blue6 = (66/255., 146/255., 198/255.)
    blue7 = (33/255., 113/255., 181/255.)
    blue8 = (8/255., 81/255., 156/255.)
    blue9 = (8/255., 48/255., 107/255.)

    purple1 = (252/255., 251/255., 253/255.)
    purple2 = (239/255., 237/255., 245/255.)
    purple3 = (218/255., 218/255., 235/255.)
    purple4 = (188/255., 189/255., 220/255.)
    purple5 = (158/255., 154/255., 200/255.)
    purple6 = (128/255., 125/255., 186/255.)
    purple7 = (106/255., 81/255., 163/255.)
    purple8 = (84/255., 39/255., 143/255.)
    purple9 = (63/255., 0/255., 125/255.)

    green1 = (247/255., 252/255., 245/255.)
    green2 = (229/255., 245/255., 224/255.)
    green3 = (199/255., 233/255., 192/255.)
    green4 = (161/255., 217/255., 155/255.)
    green5 = (116/255., 196/255., 118/255.)
    green6 = (65/255., 171/255., 93/255.)
    green7 = (35/255., 139/255., 69/255.)
    green8 = (0/255., 109/255., 44/255.)
    green9 = (0/255., 68/255., 27/255.)

    gray1 = (255/255., 255/255., 255/255.)
    gray2 = (240/255., 240/255., 240/255.)
    gray3 = (217/255., 217/255., 217/255.)
    gray4 = (189/255., 189/255., 189/255.)
    gray5 = (150/255., 150/255., 150/255.)
    gray6 = (115/255., 115/255., 115/255.)
    gray7 = (82/255., 82/255., 82/255.)
    gray8 = (37/255., 37/255., 37/255.)
    gray9 = (0/255., 0/255., 0/255.)

    red1 = (255/255., 245/255., 240/255.)
    red2 = (254/255., 224/255., 210/255.)
    red3 = (252/255., 187/255., 161/255.)
    red4 = (252/255., 146/255., 114/255.)
    red5 = (251/255., 106/255., 74/255.)
    red6 = (239/255., 59/255., 44/255.)
    red7 = (203/255., 24/255., 29/255.)
    red8 = (165/255., 15/255., 21/255.)
    red9 = (103/255., 0/255., 13/255.)

### Qualitative colors (pastels):

    q1 = (141/255., 255/255., 199/255.)  #aqua
    q2 = (255/255., 255/255., 179/255.)  #pale yellow
    q3 = (190/255., 186/255., 218/255.)  #lavender
    q4 = (251/255., 128/255., 114/255.)  #pink/orange
    q5 = (128/255., 177/255., 211/255.)  #light blue
    q6 = (253/255., 180/255., 98/255.)   #light orange
    q7 = (179/255., 222/255., 105/255.)  #lime
    q8 = (252/255., 205/255., 229/255.)  #pink
    q9 = (217/255., 217/255., 217/255.)  #light gray
    q10 = (188/255., 128/255., 189/255.) #purple
    q11 = (204/255., 235/255., 197/255.) #pale green
    q12 = (255/255., 237/255., 111/255.) #yellow

### Qualitative colors (bright):

    b1 = (228/255., 26/255., 28/255.)   #red
    b2 = (55/255., 126/255., 184/255.)  #blue
    b3 = (77/255., 175/255., 74/255.)   #green
    b4 = (152/255., 78/255., 163/255.)  #purple
    b5 = (255/255., 127/255., 0/255.)   #orange
    b6 = (255/255., 255/255., 51/255.)  #yellow
    b7 = (166/255., 86/255., 40/255.)   #brown
    b8 = (247/255., 129/255., 191/255.) #pink

### NWS Reflectivity Colors (courtesy MetPy library):

    c5 =  (0.0,                 0.9254901960784314, 0.9254901960784314)
    c10 = (0.00392156862745098, 0.6274509803921569, 0.9647058823529412)
    c15 = (0.0,                 0.0,                0.9647058823529412)
    c20 = (0.0,                 1.0,                0.0)
    c25 = (0.0,                 0.7843137254901961, 0.0)
    c30 = (0.0,                 0.5647058823529412, 0.0)
    c35 = (1.0,                 1.0,                0.0)
    c40 = (0.9058823529411765,  0.7529411764705882, 0.0)
    c45 = (1.0,                 0.5647058823529412, 0.0)
    c50 = (1.0,                 0.0,                0.0)
    c55 = (0.8392156862745098,  0.0,                0.0)
    c60 = (0.7529411764705882,  0.0,                0.0)
    c65 = (1.0,                 0.0,                1.0)
    c70 = (0.6,                 0.3333333333333333, 0.788235294117647)
    c75 = (0.0,                 0.0,                0.0) 

    
    gray1 = (255/255., 255/255., 255/255.)
    gray125 = (251/255., 251/255., 251/255.)
    gray15 = (247/255., 247/255., 247/255.)
    gray175 = (243/255., 243/255., 243/255.)
    gray2 = (240/255., 240/255., 240/255.)
    gray225 = (234/255., 234/255., 234/255.)
    gray25 = (228/255., 228/255., 228/255.)
    gray275 = (223/255., 223/255., 223/255.)
    gray3 = (217/255., 217/255., 217/255.)
    gray325 = (210/255., 210/255., 210/255.) 
    gray35 = (204/255., 204/255., 204/255.)
    gray375 = (197/255., 197/255., 197/255.)
    gray4 = (189/255., 189/255., 189/255.)
    gray425 = (180/255., 180/255., 180/255.)
    gray45 = (170/255., 170/255., 170/255.)
    gray475 = (160/255., 160/255., 160/255.)
    gray5 = (150/255., 150/255., 150/255.)
    gray525 = (140/255., 140/255., 140/255.)
    gray55 = (130/255., 130/255., 130/255.)
    gray575 = (122/255., 122/255., 122/255.)
    gray6 = (115/255., 115/255., 115/255.)
    gray625 = (107/255., 107/255., 107/255.)
    gray65 = (99/255., 99/255., 99/255.)
    gray675 = (91/255., 91/255., 91/255.)
    gray7 = (82/255., 82/255., 82/255.)
    gray725 = (69/255., 69/255., 69/255.)
    gray75 = (50/255., 50/255., 50/255.)
    gray775 = (43/255., 43/255., 43/255.)
    gray8 = (37/255., 37/255., 37/255.)
    gray825 = (25/255., 25/255., 25/255.)
    gray85 = (13/255., 13/255., 13/255.)
    gray875 = (2/255., 2/255., 2/255.)
    gray9 = (0/255., 0/255., 0/255.)
     
### GOES-13 IR TB
    ir63 = ( 0.000 , 0.000 , 0.000 )
    ir62 = ( 0.063 , 0.063 , 0.063 )
    ir61 = ( 0.141 , 0.141 , 0.141 )
    ir60 = ( 0.208 , 0.208 , 0.208 )
    ir59 = ( 0.286 , 0.286 , 0.286 )
    ir585 = ( 0.31 , 0.31 , 0.31 )
    ir58 = ( 0.349 , 0.349 , 0.349 )
    ir57 = ( 0.427 , 0.427 , 0.427 )
    ir56 = ( 0.490 , 0.490 , 0.490 )
    ir555 = ( 0.52 , 0.52 , 0.52 )
    ir55 = ( 0.573 , 0.573 , 0.573 )
    ir545 = ( 0.6 , 0.6 , 0.6 )
    ir54 = ( 0.635 , 0.635 , 0.635 )
    ir535 = ( 0.67 , 0.67 , 0.67 )
    ir53 = ( 0.714 , 0.714 , 0.714 )
    ir525 = ( 0.74 , 0.74 , 0.74 ) 
    ir52 = ( 0.776 , 0.776 , 0.776 )
    ir515 = ( 0.81 , 0.81 , 0.81 )
    ir51 = ( 0.85 , 0.85 , 0.85 )
    ir50 = ( 0.88 , 0.88 , 0.88 )
    ir49 = ( 0.91 , 0.91 , 0.91 )
    ir48 = ( 0.000 , 0.714 , 1.000 )
    ir47 = ( 0.000 , 0.651 , 0.953 )
    ir46 = ( 0.000 , 0.588 , 0.906 )
    ir45 = ( 0.000 , 0.541 , 0.859 )
    ir44 = ( 0.000 , 0.475 , 0.808 )
    ir43 = ( 0.000 , 0.412 , 0.761 )
    ir42 = ( 0.000 , 0.365 , 0.714 )
    ir41 = ( 0.000 , 0.302 , 0.667 )
    ir40 = ( 0.000 , 0.239 , 0.620 )
    ir39 = ( 0.000 , 0.192 , 0.573 )
    ir38 = ( 0.000 , 0.125 , 0.525 )
    ir37 = ( 0.000 , 0.063 , 0.475 )
    ir36 = ( 0.000 , 0.239 , 0.604 )
    ir35 = ( 0.000 , 0.365 , 0.475 )
    ir34 = ( 0.000 , 0.475 , 0.365 )
    ir33 = ( 0.000 , 0.604 , 0.239 )
    ir32 = ( 0.000 , 0.729 , 0.110 )
    ir31 = ( 0.000 , 0.843 , 0.000 )
    ir30 = ( 0.000 , 1.000 , 0.000 )
    ir29 = ( 0.141 , 1.000 , 0.000 )
    ir28 = ( 0.286 , 1.000 , 0.000 )
    ir27 = ( 0.427 , 1.000 , 0.000 )
    ir26 = ( 0.573 , 1.000 , 0.000 )
    ir25 = ( 0.714 , 1.000 , 0.000 )
    ir24 = ( 0.859 , 1.000 , 0.000 )
    ir23 = ( 1.000 , 1.000 , 0.000 )
    ir22 = ( 1.000 , 0.906 , 0.000 )
    ir21 = ( 1.000 , 0.808 , 0.000 )
    ir20 = ( 1.000 , 0.714 , 0.000 )
    ir19 = ( 1.000 , 0.635 , 0.000 )
    ir18 = ( 1.000 , 0.541 , 0.000 )
    ir17 = ( 1.000 , 0.443 , 0.000 )
    ir16 = ( 1.000 , 0.349 , 0.000 )
    ir15 = ( 1.000 , 0.271 , 0.000 )
    ir14 = ( 1.000 , 0.176 , 0.000 )
    ir13 = ( 1.000 , 0.078 , 0.000 )
    ir12 = ( 1.000 , 0.000 , 0.000 )
    ir11 = ( 0.000 , 0.000 , 0.000 )
    ir10 = ( 0.078 , 0.078 , 0.078 )
    ir9 = ( 0.176 , 0.176 , 0.176 )
    ir8 = ( 0.271 , 0.271 , 0.271 )
    ir7 = ( 0.349 , 0.349 , 0.349 )
    ir6 = ( 0.443 , 0.443 , 0.443 )
    ir5 = ( 0.541 , 0.541 , 0.541 )
    ir4 = ( 0.635 , 0.635 , 0.635 )
    ir3 = ( 0.714 , 0.714 , 0.714 )
    ir2 = ( 0.808 , 0.808 , 0.808 )
    ir1 = ( 0.906 , 0.906 , 0.906 )
    ir0 = ( 0.984 , 0.984 , 0.984 )

    hail_ml_cmap = matplotlib.colors.ListedColormap([blue2, blue3, blue4, red2, 
                    red3, red4, red5, purple6, purple5, purple4, purple3])
    
    cin_cmap = matplotlib.colors.ListedColormap([purple7, purple6, purple5, 
              purple4, blue4, blue3, blue2, blue1])

    wz_cmap = matplotlib.colors.ListedColormap([blue2, blue3, blue4, red2, red3, 
             red4, red5, red6, red7])

    dz_cmap_2 = matplotlib.colors.ListedColormap([blue5, blue3, green3, green5, 
               green7, orange3, orange5, orange7, red7, red8, purple8, purple6])

    dz_cmap = matplotlib.colors.ListedColormap([green5, green4, green3, orange2, 
             orange4, orange6, red6, red4, purple3, purple5])

    nws_dz_cmap = matplotlib.colors.ListedColormap([c20, c25, c30, c35, c40, c45, 
                 c50, c55, c60, c65, c70])

    nws_dz_cmap_clear_air = matplotlib.colors.ListedColormap([c5, c10, c15, c20, 
                           c25, c30, c35, c40, c45, c50, c55, c60, c65, c70])

    wind_svr_cmap = matplotlib.colors.ListedColormap([gray2, gray3, blue1, 
                    blue2, blue3, blue4, blue5, orange3, orange4, 
                    orange5, red4, red5, red6, red7, red8, purple8, 
                    purple7, purple6])

    wind_trop_cmap = matplotlib.colors.ListedColormap([gray2, gray3, blue1, 
                    blue2, blue3, blue4, blue5, orange3, orange4, orange5, 
                    orange6, orange7, red3, red4, red5, red6, red7, red8, red9, 
                    purple8, purple7])

    wind_cmap = matplotlib.colors.ListedColormap([gray1, gray2, gray3, orange2, 
               orange3, orange4, orange5, orange6, red7, red8])

    wz_cmap_extend = matplotlib.colors.ListedColormap([blue2, blue3, blue4, red2, 
                    red3, red4, red5, red6, red7, purple7, purple6, purple5])

    cape_cmap = matplotlib.colors.ListedColormap([blue2, blue3, blue4, orange2, 
               orange3, orange4, orange5, red4, red5, red6, red7, purple7, 
               purple6, purple5])

    td_cmap_ncar = matplotlib.colors.ListedColormap(['#ad598a', '#c589ac', 
                  '#dcb8cd', '#e7cfd1', '#d0a0a4', '#ad5960', '#8b131d', 
                  '#8b4513', '#ad7c59', '#c5a289', '#dcc7b8', '#eeeeee', 
                  '#dddddd', '#bbbbbb', '#e1e1d3', '#e1d5b1', '#ccb77a', 
                  '#ffffe5', '#f7fcb9', '#addd8e', '#41ab5d', '#006837', 
                  '#004529', '#195257', '#4c787c'])

    temp_cmap_ugly = matplotlib.colors.ListedColormap([blue6, blue4, blue2, 
                    green6, green4, green2, orange2, orange4, red5, red7, purple7])

    temp_cmap = matplotlib.colors.ListedColormap([purple4, purple5, purple6, 
               purple7, blue8, blue7, blue6, blue5, blue4, blue3, green7, 
               green6, green5, green4, green3, green2, orange2, orange3, 
               orange4, orange5, red5, red6, red7, red8, purple6, purple5, 
               purple4, purple3])

    blues_cmap = matplotlib.colors.ListedColormap([blue3, blue4, blue5, blue6, 
                blue7])

    oranges_cmap = matplotlib.colors.ListedColormap([orange3, orange4, orange5, 
                  orange6, orange7])

    td_cmap_ugly = matplotlib.colors.ListedColormap([orange3, gray4, gray3, 
                  gray1, green3, green5, green7, blue3, blue5, blue7, purple3])

    td_cmap = matplotlib.colors.ListedColormap([gray6, gray5, gray4, gray3, 
             gray2, gray1, green1, green2, green3, green4, green5, green6, 
             blue3, blue4, blue5, purple3])

    uv_cmap = matplotlib.colors.ListedColormap([purple5, purple4, purple3, 
             purple2, purple1, orange1, orange2, orange3, orange4, orange5])

    diff_cmap = matplotlib.colors.ListedColormap([blue7, blue6, blue5, blue4, 
               blue3, blue2, blue1, red1, red2, red3, red4, red5, red6, red7])

    mslp_cmap = matplotlib.colors.ListedColormap([purple7, purple6, purple5, 
               red7, red6, red5,orange7, orange6, orange5, green8, green7, 
               green6, blue6, blue5, blue4, gray3, gray2, gray1])

    paintball_colors = matplotlib.colors.ListedColormap([q1, b8, q3, q4, q5, q6, 
                      q7, q8, b6, q10, q11, b3, b2, purple5, red5, green5, 
                      blue5, orange5])

    paintball_colors_list = [q1, b8, q3, q4, q5, q6, q7, q8, b6, q10, q11, b3, 
                           b2, purple5, red5, green5, blue5, orange5]

    mslp_paint_colors = matplotlib.colors.ListedColormap([purple8,purple6, 
                       purple4, red8, red6, red4, orange8, orange6, orange4, 
                       green8, green6, green4, blue8, blue6, blue4, gray3, 
                       gray2, gray1])

    all_blues_cmap =  matplotlib.colors.ListedColormap([blue1, blue2, blue3, 
                     blue4, blue5, blue6, blue7, blue8, blue9])

    all_greens_cmap =  matplotlib.colors.ListedColormap([green1, green2, green3, 
                      green4, green5, green6, green7, green8, green9])

    all_reds_cmap =  matplotlib.colors.ListedColormap([red1, red2, red3, red4, 
                    red5, red6, red7, red8, red9])

    mfc_cmap = matplotlib.colors.ListedColormap([gray1, orange2, orange3, 
              orange4, red5, red6, red7])

    corf_cmap = matplotlib.colors.ListedColormap([purple7, purple5, purple3, 
               purple1, gray1, gray1, orange1, orange3, orange5, orange7])

    ul_dvg_cmap = matplotlib.colors.ListedColormap([green8, green6, green4, 
                 green2, gray1, gray1, purple2, purple4, purple6, purple8])

    rain_cmap = matplotlib.colors.ListedColormap([green2, green3, green5, 
               blue4, blue5, blue6, purple6, purple5, purple4, red4, red5, red6])

    cp_cmap   = matplotlib.colors.ListedColormap([gray8, gray6, purple6, 
               purple5, purple4, blue7, blue6, blue5, blue4, green6, green5, 
               green4, green3, orange6, orange5, orange4, red5, red6, red8])

    cwp_cmap = matplotlib.colors.ListedColormap([gray2, gray3, gray4, gray5, 
              gray6, blue6, blue5, blue4, blue3, green6, green5, green4, 
              green3, green2, orange2, orange3, orange4, orange5, red5, red6, 
              red7, red8, purple6, purple7, purple8])

    mslp_cmap = matplotlib.colors.ListedColormap([purple8, purple6, purple4, 
               red8, red6, red4, orange8, orange6, orange4, green8, green6, 
               green4, blue8, blue6, blue4, gray3, gray2, gray1])

    pw_cmap = matplotlib.colors.ListedColormap([orange8, orange7, orange6, orange5, orange4, orange3, 
             orange2, green1, green2, green3, green4, green5, green6, green7, 
             green8, green9, blue4, blue5, blue6, blue7, blue8, purple4, 
             purple5, purple6, purple7, red4, red5, red6, red7, red8])

    omega_cmap = matplotlib.colors.ListedColormap([purple7, purple5, purple3, 
                purple2, gray1, green2, green3, green5, green7])

    sw_cmap = matplotlib.colors.ListedColormap([blue9, blue8, blue7, blue6, blue5, 
             blue4, green8, green7, green6, green5, orange4, orange5, orange6, 
             orange7, red5, red6, red7, red8, purple7])

    fed_cmap = matplotlib.colors.ListedColormap([blue2, blue4, blue6, orange2, 
               orange4, orange6, red2, red4, red6, purple3, 
               purple5, purple7])

    echo_tops_cmap = cape_cmap

    rh_cmap = matplotlib.colors.ListedColormap([red8, red7, red6, red5, orange6, \
                 orange5, orange4, orange3, gray6, gray5, gray4, gray3, gray2, gray1, \
                 blue2, blue3, blue4, blue5, blue6])
    
    rfti_cmap = matplotlib.colors.ListedColormap([blue6, blue4, green6, green4, \
                 orange4, orange6, red4, red6, purple4])

    frp_cmap = matplotlib.colors.ListedColormap([blue8, green8, orange8, red8])

    smoke_cmap = matplotlib.colors.ListedColormap([c5, c10, c15, c20, c25, c30, c35, c40, c45, c50, c55, c60, c65, c70])               

    smokeh_cmap = matplotlib.colors.ListedColormap([gray2, gray3, blue1, blue2, blue3, green2, green3, green4, orange3, orange4, orange5, red4, red6])
    # Hardcodedat the moment. 
    paint_cmap = matplotlib.colors.ListedColormap([gray8, gray8])
    pmm_dz_colors_gray = [gray7, gray9]

    
    gray_paintball_color = gray5 
    
    
    # Satellite Colormaps =====================================================

    ir_cmap = matplotlib.colors.ListedColormap([ir6,ir7,ir8,ir9,ir10,
                                                ir11,purple9,purple8,
                                                purple7,ir12,ir13,ir14,ir15,ir16,
                                                ir17,ir18,ir19,ir20,ir21,ir22,ir23,
                                                ir24,ir25,ir26,ir27,
                                               ir28,ir29,ir30,ir31,ir32,ir33,ir34,
                                                ir35,ir36,ir37,ir38,ir39,
                                                ir40,ir41,ir42,ir43,ir44,ir45,
                                               ir46,ir47,ir48,ir49,ir50,ir51,ir515, ir52,ir525,ir53,ir535,
                                                ir54,ir545,ir55,ir555,ir56,ir57,ir58,ir585,ir59,
                                               ir60,ir61,ir62])   

    wv_cmap = matplotlib.colors.ListedColormap([green8, green8, green7, green7, green6, 
                                                green5, green4, blue3, blue4, blue5, blue5, blue6, 
                                                blue7, blue8, blue9, purple7, purple6, purple5, purple4, 
                                                gray2, gray3, gray4, gray5, gray5, gray6, gray7, 
                                                gray8, gray9,orange5,orange5,orange6,orange7,
                                                red4,red5, red6, red7, red8,red8, red9,red9,red9 ])

    vis_cmap = matplotlib.colors.ListedColormap([gray9,gray9,gray875,gray85,gray825,
                                               gray8,gray775,gray75,gray75,gray725,gray7,
                                               gray675,gray65,gray65,gray625,gray6,gray575,
                                               gray55,gray55,gray525,gray5,gray475,gray45,gray45,
                                               gray425,gray4,gray4,gray375,gray35,gray325,gray3,
                                               gray275,gray25,gray225,gray2,gray175,gray15,gray125,gray1,gray1 ] )
    
    irbw_cmap = matplotlib.colors.ListedColormap([gray1, gray15, gray2, gray25, gray3,
                                                  gray35, gray4, gray45, gray5, gray55, 
                                                  gray6, gray65, gray7, gray725, gray75, 
                                                  gray8, gray85, gray9] )

    
    
    
