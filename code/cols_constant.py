DF_COLS = ["Sh_Blocks",
           "Blocks_Blocks",
           "Pass_Blocks",
           "Tkl_Tackles",
           "Tkl_Vs_Dribbles",
           "Att_Vs_Dribbles",
           "Past_Vs_Dribbles",
           "Def 3rd_Tackles",
           "Int",
           "Mid 3rd_Tackles"]

DF_DB_COLS = ["Dead_Pass_Types",
              "Att 3rd_Touches",
              "PPA",
              "Def 3rd_Touches",
              "Cmp_percent_Medium",
              "Sh_Blocks",
              "Prog_Receiving",
              "CrsPA"]

DF_CB_COLS = ["Cmp_Long",
              "Tkl+Int",
              "Clr",
              "Sh_Blocks",
              "Blocks_Blocks",
              "Cmp_Total",
              "Def 3rd_Tackles",
              "Att_Total",
              "Prog"]
MF_COLS = ["Live_Pass_Types",
           "Tkl+Int",
           "Att_Total",
           "Def_GCA_Types",
           "Fld_GCA_Types",
           "Tkl_Tackles",
           "xA",
           "Drib_GCA_Types"]

MF_CM_COLS = ["Prog_Receiving",
              "Att 3rd_Touches",
              "Mis_Dribbles",
              "Final_Third",
              "Def 3rd_Touches",
              "Int",
              "Mid 3rd_Touches"]

MF_DM_COLS = ["Att_Long",
              "Prog_Receiving",
              "Tkl+Int",
              "Final_Third",
              "Mis_Dribbles",
              "Clr",
              "Int",
              "Sh_Blocks",
              "Att_Medium",
              "PassLive_GCA_Types"]

MF_AM_COLS = ["Att 3rd_Touches",
              "SoT_Standard",
              "SCA_SCA",
              "Prog_Receiving",
              "CK_Pass_Types",
              "GCA_GCA",
              "PPA",
              "PassDead_SCA_Types",
              "Clr"]

FW_COLS = ["xG_Expected",
           "PKatt_Standard",
           "Drib_GCA_Types",
           "PassDead_GCA_Types",
           "PassLive_GCA_Types",
           "GCA_GCA",
           "Sh_GCA_Types",
           "Past_Vs_Dribbles"]

FW_AW_COLS = ["Mis_Dribbles",
              "Att Pen_Touches",
              "Drib_SCA_Types",
              "Final_Third",
              "Sh_Standard",
              "KP",
              "Mid 3rd_Tackles"]

FW_ST_COLS = ["Att_Medium",
              "Gls_Standard",
              "PrgDist_Total",
              "Sh_Standard",
              "xG_Expected",
              "GCA_GCA"]

SHOOTING = ["Gls_Standard","Sh_Standard","SoT_Standard","G_per_Sh_Standard","G_per_SoT_Standard","Dist_Standard","xG_Expected","npxG_per_Sh_Expected"]
PASSING = ["Cmp_Total","Att_Total","TotDist_Total","PrgDist_Total","Cmp_Short","Att_Short","Cmp_Medium","Att_Medium","Cmp_Long","Att_Long","Ast","xAG","xA","KP","Final_Third","PPA","CrsPA","Prog"]
GSCREATION = ["SCA_SCA","PassLive_SCA_Types","PassDead_SCA_Types","Drib_SCA_Types","Sh_SCA_Types","Fld_SCA_Types","Def_SCA_Types","GCA_GCA","PassLive_GCA_Types","PassDead_GCA_Types","Drib_GCA_Types","Sh_GCA_Types","Fld_GCA_Types","Def_GCA_Types"]
GSCREATION_MEAN = ["SCA_SCA","GCA_GCA"]
DEFENSIVE = ["Tkl_Tackles","TklW_Tackles","Tkl_Vs_Dribbles","Blocks_Blocks","Sh_Blocks","Pass_Blocks","Int","Tkl+Int","Clr"]
POSSESSION = ["Touches_Touches","Succ_Dribbles","Att_Dribbles","Rec_Receiving","Prog_Receiving"]
