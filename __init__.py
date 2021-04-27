# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

bl_info = {
     "name": "X-Slicer",
     "author": "Jimmy",
     "version": (0, 9, 0),
     "blender": (2, 91, 0),
     "location": "3D View > Tools Panel",
     "description": "Makes a series of cross-sections and exports an svg file",
     "warning": "",
     "wiki_url": "tba",
     "tracker_url": "https://github.com/aenckli/x_slicer",
     "category": "Object"}

import bpy, os, bmesh, numpy
from bpy.props import FloatProperty, BoolProperty, EnumProperty, IntProperty, StringProperty, FloatVectorProperty
from mathutils import Vector
import math
from itertools import groupby

# global constants
DELTA = 1e-6
INFINITY_POS = 100
INFINITY_NEG = -100

def newrow(layout, s1, root, s2, e):
    row = layout.row()
    row.label(text = s1)
    row.prop(root, s2)
    row.enabled = e

def interpolate_scalar(x1, y1, x2, y2, inter_value):
    return y1 if abs(x2-x1) < DELTA else (inter_value - x1)*(y2 - y1)/(x2 - x1) + y1

# get vertice indices within boundary
def getBoundaryVertices(vt, idx_midline, x1, y1, x2, y2, axis_x, axis_y, sp_st, sp_end):
    inside = [x for x in vt[sp_st:sp_end+1] if x != -1 and ((x[axis_x] >= x1 and x[axis_y] >= y1) and (x[axis_x] <= x2 and x[axis_y] <= y2))]

    inside_idx = [vt.index(x) for x in inside]
    inside_idx.sort()

    idx_1 = idx_midline
    idx_0 = sp_end if idx_1 == sp_st else idx_1-1
    idx_2 = sp_st if idx_1 == sp_end else idx_1+1

    idx = [idx_1, idx_2, idx_1, idx_2]

    if len(inside_idx) == 1:
        if inside_idx[0] ==  idx_1:
            idx[0] = idx_0
            idx[1] = idx_1
        elif inside_idx[0] == idx_2:
            idx[2] = idx_2
            idx[3] = 0 if idx_2 == sp_end else idx_2+1
    elif len(inside_idx) > 1:
        gb = groupby(enumerate(inside_idx), key=lambda x: x[0] - x[1])
        all_groups = ([i[1] for i in g] for _, g in gb)
        c = list(filter(lambda x: len(x) > 0, all_groups))
        if idx_1 in c[-1] or idx_2 in c[-1]:
            d = [c[-1]]
            if sp_st in c[0]:
                d.insert(0, c[0])
        elif idx_1 in c[0] or idx_2 in c[0]:
            d = [c[0]]
            if sp_end in c[-1]:
                d.insert(1, c[-1])
        else:
            d = [x for x in c if idx_1 in x or idx_2 in x]

        if len(d) > 0:
            idx[1] = min(d[1]) if len(d) > 1 else min(d[0])
            idx[2] = max(d[0])
            idx[0] = sp_end if idx[1] == sp_st else idx[1]-1
            idx[3] = sp_st if idx[2] == sp_end else idx[2]+1

    return idx

   
def slicer(settings):
    nameaxis = ['X', 'Y', 'Z']
    nameslice = ['SlicesX', 'SlicesY', 'SlicesZ']
    f_scale = 1000 * bpy.context.scene.unit_settings.scale_length

    dp = bpy.context.evaluated_depsgraph_get()
    aob = bpy.context.active_object
    tempmesh = aob.evaluated_get(dp).to_mesh()
    omw = aob.matrix_world
    bm = bmesh.new()
    bm.from_mesh(tempmesh)
    bmesh.ops.transform(bm, matrix=omw, space=bpy.context.object.matrix_world, verts=bm.verts)
    aob.evaluated_get(dp).to_mesh_clear()
    aob.select_set(False)

    mwidth = settings.x_slicer_material_width
    mheight = settings.x_slicer_material_height
    lt = settings.x_slicer_material_thick/f_scale

    ndir = int(settings.x_slicer_cut_ndir)
    acut = [int(settings.x_slicer_cut_plane), int(settings.x_slicer_cut_plane_2)]
    ls = [settings.x_slicer_cut_spacing/f_scale, settings.x_slicer_cut_spacing_2/f_scale]
    ccen = [settings.x_slicer_cut_center, settings.x_slicer_cut_center_2]
    lfirst = [settings.x_slicer_cut_first_loc/f_scale, settings.x_slicer_cut_first_loc_2/f_scale]
    minz = [min([v.co[acut[0]] for v in bm.verts]), min([v.co[acut[1]] for v in bm.verts])]
    maxz = [max([v.co[acut[0]] for v in bm.verts]), max([v.co[acut[1]] for v in bm.verts])]
    csides = [(settings.x_slicer_cut_sides, 1)[minz[0] * maxz[0] >= 0], (settings.x_slicer_cut_sides_2, 1)[minz[1] * maxz[1] >= 0]]
    lh = [((0, minz[0])[csides[0] == 1] + lfirst[0] + lt*0.5, 0)[csides[0] == 2 and ccen[0] == True], ((0, minz[1])[csides[1] == 1] + lfirst[1] + lt*0.5, 0)[csides[1] == 2 and ccen[1] == True]]

    sepfile = settings.x_slicer_separate_files
    ofile = settings.x_slicer_ofile
    ct = settings.x_slicer_cut_thickness/f_scale
    svgpos = settings.x_slicer_svg_position
    dpi = settings.x_slicer_dpi
    lcol = settings.x_slicer_cut_colour
    lthick = settings.x_slicer_cut_line

    mm2pi = dpi/25.4
    scale = f_scale*mm2pi

    vtlist = []

# ---------------------------------------------------------------------------------------------------------------
# CREATE SLICE OBJECT AND FILL MESH WITH SLICED VERTICES
#   EXTRACT SLICE VERTICES
    for idir in range(ndir):
        
# ---------------------------------------------------------------------------------------------------------------
        vlen, elen, vlenlist, elenlist = 0, 0, [0], [0]
        vtlist.append([])
        vlist = []
        elist = []
        erem = []

        vtlist_mesh = []
        edlist_mesh = []
    
        side = 0
        endthickness = maxz[idir]
        vecAxis = [(1, 0.0, 0.0), (0.0, 1, 0.0), (0.0, 0.0, 1)]
        vecZero = [0.0, 0.0, 0.0]
        
        while (side < csides[idir]):
            while (lh[idir] < endthickness and side == 0) or (lh[idir] > endthickness and side == 1):
                cbm = bm.copy()
                vecZero[acut[idir]] = lh[idir]
                newgeo = bmesh.ops.bisect_plane(cbm, geom = cbm.edges[:] + cbm.faces[:], dist = 0, plane_co = tuple(vecZero), plane_no = vecAxis[acut[idir]], clear_outer = False, clear_inner = False)['geom_cut']
                newverts = [v for v in newgeo if isinstance(v, bmesh.types.BMVert)]        
                newedges = [e for e in newgeo if isinstance(e, bmesh.types.BMEdge)]        
                voffset = min([v.index for v in newverts])
                
                temp = [(v.co[0], v.co[1], v.co[2]) for v in newverts]  
                vtlist_mesh.extend(temp)
                
                temp = [[v.index  - voffset + vlen for v in e.verts] for e in newedges]
                edlist_mesh.extend(temp)

                vlen += len(newverts)
                elen += len(newedges)
                vlenlist.append(len(newverts) + vlenlist[-1])
                elenlist.append(len(newedges) + elenlist[-1])

                lh[idir] += (-(lt+ls[idir]), (lt+ls[idir]))[side == 0]
                cbm.free()  
                
            side += 1
            endthickness = minz[idir]
            lh[idir] = (-(lfirst[idir] + lt*0.5), -(ls[idir] + lt))[ccen[idir] == True]
            
        if (idir >= ndir - 1):
            bm.free()        

# ---------------------------------------------------------------------------------------------------------------
# REORDER RANDOM INDEXED VERTICES
        
        vranges = [(vlenlist[i], vlenlist[i+1], elenlist[i], elenlist[i+1]) for i in range(len(vlenlist) - 1)]
        vtlist[idir] = []

        for vr in vranges:
            vlist, elist, erem = [], [], []
            layeredges = edlist_mesh[vr[2]:vr[3]]

            edgeverts = [ed[0] for ed in layeredges] + [ed[1] for ed in layeredges]
            edgesingleverts = [ev for ev in edgeverts if edgeverts.count(ev) == 1]

            for ed in layeredges:
                el = [ev for ev in edgeverts if edgeverts.count(ev) > 2]
                if ed[0] in el and ed[1] in el:
                    erem.append(ed)
            for er in erem:
                layeredges.remove(er)

            if edgesingleverts:
                e = [ed for ed in layeredges if ed[0] in edgesingleverts or ed[1] in edgesingleverts][0]
                if e[0] in edgesingleverts:
                    vlist.append(e[0])
                    vlist.append(e[1])
                else:
                    vlist.append(e[1])
                    vlist.append(e[0])
                elist.append(e)
            else:   
                elist.append(layeredges[0])
                vlist.append(elist[0][0])
                vlist.append(elist[0][1])
                
            while len(elist) < len(layeredges):
                va = 0
                for e in [ed for ed in layeredges if ed not in elist]:
                    if e[0] not in vlist and e[1] == vlist[-1]:
                        va = 1
                        vlist.append(e[0])
                        elist.append(e)
                        
                        if len(elist) == len(layeredges):
                           vlist.append(-2)
                            
                    if e[1] not in vlist and e[0] == vlist[-1]:
                        va = 1
                        vlist.append(e[1])
                        elist.append(e)
                         
                        if len(elist) == len(layeredges):
                           vlist.append(-2)
                            
                    elif e[1] in vlist and e[0] in vlist and e not in elist:
                        elist.append(e)
                        va = 2
                                                                                
                if va in (0, 2):
                    vlist.append((-1, -2)[va == 0])
                    
                    if len(elist) < len(layeredges):
                        try:
                            e1 = [ed for ed in layeredges if ed not in elist and (ed[0] in edgesingleverts or ed[1] in edgesingleverts)][0]
                            if e1[0] in edgesingleverts:
                                vlist.append(e1[0])
                                vlist.append(e1[1])
                            else:
                                vlist.append(e1[1])
                                vlist.append(e1[0])
                                
                        except Exception as e:
                            e1 = [ed for ed in layeredges if ed not in elist][0]
                            vlist.append(e1[0])
                            vlist.append(e1[1])
                        elist.append(e1)
    
            vtlist[idir].append([(vtlist_mesh[v], v)[v < 0] for v in vlist])            
            
            i = 1
            last_i = 0
            sp_st = 0
            while i < len(vtlist[idir][-1]):
                if vtlist[idir][-1][i] == -1:
                    v = vtlist[idir][-1][sp_st]
                    vl = vtlist[idir][-1][last_i]
                    sp_st = i + 1
                    if abs(v[0] - vl[0]) < DELTA and abs(v[1] - vl[1]) < DELTA and abs(v[2] - vl[2]) < DELTA:
                        vtlist[idir][-1].pop(last_i)
                        sp_st = last_i + 1
                        continue
                    last_i -= 1
                else:        
                    if vtlist[idir][-1][i-1] != -1:
                        v = vtlist[idir][-1][i]
                        vl = vtlist[idir][-1][last_i]
                        if abs(v[0] - vl[0]) < DELTA and abs(v[1] - vl[1]) < DELTA and abs(v[2] - vl[2]) < DELTA:
                            vtlist[idir][-1].pop(i)
                            continue
                    last_i = i

                i += 1

# ---------------------------------------------------------------------------------------------------------------
# DEVELOP SLOTS

    if (ndir > 1):

        free_axis = list({0,1,2}.difference({acut[0], acut[1]}))[0]
        nlayers = [len(vtlist[0]), len(vtlist[1])]

        vl = [[],[]]
        level = [0,0]

        for i in range(nlayers[1]):
            vl[1] = vtlist[1][i]
            level[1] = vl[1][0][acut[1]]

            for j in range(nlayers[0]):
                vl[0] = vtlist[0][j]
                level[0] = vl[0][0][acut[0]]

                slot_pt_list = [[],[]]
                slot_pt_idx_list = [[],[]]
                slot_bk_list = [[],[]]
                slot_block_list = [[],[]]
                cut_pt_list = []

                for l in range(2):
                    slot_pt_list[l].append([])        
                    slot_pt_idx_list[l].append([])
                    idx_spline_st = 0
                    ispline = 0
                    spline_add = False
                    k = 0
                    while (k < len(vl[l])):
                        pt_inter = [0,0,0]
                        pt_inter[acut[0]] = level[0]
                        pt_inter[acut[1]] = level[1]

                        pt1 = vl[l][k]
                        x1 = pt1[acut[l^1]]
                        if vl[l][k+1] == -1:
                            pt2 = vl[l][idx_spline_st]
                            spline_add = k+1 < len(vl[l])
                        else:
                            pt2 = vl[l][k+1]
                        x2 = pt2[acut[l^1]]
                        lev = level[l^1]
                        if (x1 < lev and x2 >= lev) or (x1 > lev and x2 <= lev):
                            pt_inter[free_axis] = interpolate_scalar(x1, pt1[free_axis], x2, pt2[free_axis], lev)
                            slot_pt_list[l][ispline].append(pt_inter)
                            slot_pt_idx_list[l][ispline].append(k)
                        if spline_add ==  True:
                            ispline += 1
                            idx_spline_st = k+2
                            slot_pt_list[l].append([])
                            slot_pt_idx_list[l].append([])
                            spline_add = False
                        if vl[l][k+1] == -1:
                            k += 1
                        k += 1

                for c in range(2):
                    for k in range(len(slot_pt_list[c])):         # spline
                        for l in range(len(slot_pt_list[c][k])):  # vertice
                            slot_bk_list[c].append([slot_pt_list[c][k][l][free_axis],k,l])
                    slot_bk_list[c].sort()
                
                for c in range(2):
                    k = 0
                    while k < len(slot_bk_list[c]):
                        blk_start = k
                        blk_end = k
                        for l in range(len(slot_bk_list[c])-1,-1,-1):
                            if slot_bk_list[c][l][1] == slot_bk_list[c][k][1]:
                                blk_end = l
                                slot_block_list[c].append([blk_start, blk_end])
                                k = l
                                break
                        k += 1

                idx_combined = list(set([x[0] for x in slot_block_list[0]]).union(set([x[0] for x in slot_block_list[1]])))
                idx_combined.sort()
                
                if len(slot_bk_list[0]) == 0 or len(slot_bk_list[1]) == 0:
                    continue
                    
                vt_idx_added = [[],[]]

                for k in range(len(idx_combined)):
                    idx = idx_combined[k]

                    sp_idx = [[slot_bk_list[0][idx][1], slot_bk_list[0][idx+1][1]], [slot_bk_list[1][idx][1], slot_bk_list[1][idx+1][1]]]
                    vt_idx = [[slot_bk_list[0][idx][2], slot_bk_list[0][idx+1][2]], [slot_bk_list[1][idx][2], slot_bk_list[1][idx+1][2]]]
                    vertex1 = [(x + y)/2.0 for (x, y) in zip(slot_pt_list[0][sp_idx[0][0]][vt_idx[0][0]], slot_pt_list[0][sp_idx[0][1]][vt_idx[0][1]])]
                    vertex2 = [(x + y)/2.0 for (x, y) in zip(slot_pt_list[1][sp_idx[1][0]][vt_idx[1][0]], slot_pt_list[1][sp_idx[1][1]][vt_idx[1][1]])]
                    cut_pt_list.append([(x + y)/2.0 for (x, y) in zip(vertex1, vertex2)])
                    
#   CREATE SLOTS ------------------------------------    

                    for l in range(2):
                        x = level[l^1]
                        
                        idx_add = sum(x[1] for x in vt_idx_added[l] if x[0] < slot_pt_idx_list[l][sp_idx[l][l]][vt_idx[l][l]])
                            
                        idx_1 = slot_pt_idx_list[l][sp_idx[l][l]][vt_idx[l][l]]+idx_add
                        x1 = vl[l][idx_1][acut[l^1]]
                        
                        spline_st = idx_1
                        while (spline_st != 0 and vl[l][spline_st] != -1): spline_st -= 1
                        spline_st = (spline_st+1, 0)[spline_st == 0]
                        spline_end = idx_1
                        while (spline_end < len(vl[l]) and vl[l][spline_end] != -1): spline_end += 1
                        spline_end -= 1
                        
                        dir = True if x > x1 else False

                        x_minus = x - lt/2.0
                        x_plus = x + lt/2.0

                        if l == 0:
                            idx_h = getBoundaryVertices(vl[l], idx_1, INFINITY_NEG, x_minus, cut_pt_list[k][free_axis], x_plus, free_axis, acut[l^1], spline_st, spline_end)
                        else:
                            idx_h = getBoundaryVertices(vl[l], idx_1, cut_pt_list[k][free_axis], x_minus, INFINITY_POS, x_plus, free_axis, acut[l^1], spline_st, spline_end)
                        
                        pt_minus = [(vl[l][idx_h[0]][acut[l^1]], vl[l][idx_h[0]][free_axis]), (vl[l][idx_h[1]][acut[l^1]], vl[l][idx_h[1]][free_axis])]
                        pt_plus = [(vl[l][idx_h[2]][acut[l^1]], vl[l][idx_h[2]][free_axis]), (vl[l][idx_h[3]][acut[l^1]], vl[l][idx_h[3]][free_axis])]
                        x_inter = [x_minus, x_plus] if dir else [x_plus, x_minus]
                        y_h = [0,0]
                        y_h[0] = interpolate_scalar(pt_minus[0][0], pt_minus[0][1], pt_minus[1][0], pt_minus[1][1], x_inter[0])
                        y_h[1] = interpolate_scalar(pt_plus[0][0], pt_plus[0][1], pt_plus[1][0], pt_plus[1][1], x_inter[1])
                        
                        if l == 0:
                            idx_v = getBoundaryVertices(vl[l], idx_1, INFINITY_NEG, INFINITY_NEG, cut_pt_list[k][free_axis], INFINITY_POS, free_axis, acut[l^1], spline_st, spline_end)
                        else:
                            if y_h[0] < cut_pt_list[k][free_axis]:
                                y_h[0] = vl[l][idx_1][free_axis]
                                idx_h[0] = idx_1
                                idx_h[1] = (idx_1+1, spline_st)[idx_1 == spline_end]
                            idx_v = getBoundaryVertices(vl[l], idx_1, cut_pt_list[k][free_axis], INFINITY_NEG, INFINITY_POS, INFINITY_POS, free_axis, acut[l^1], spline_st, spline_end)

                        y_v = [0,0]
                       
                        if l == 1 and idx_v[0] == spline_st-1:
                            y_v[0] = INFINITY_POS
                            y_v[1] = INFINITY_POS
                        else:
                            pt_minus = [(vl[l][idx_v[0]][free_axis], vl[l][idx_v[0]][acut[l^1]]), (vl[l][idx_v[1]][free_axis], vl[l][idx_v[1]][acut[l^1]])]
                            pt_plus = [(vl[l][idx_v[2]][free_axis], vl[l][idx_v[2]][acut[l^1]]), (vl[l][idx_v[3]][free_axis], vl[l][idx_v[3]][acut[l^1]])]
                            y_v[0] = interpolate_scalar(pt_minus[0][0], pt_minus[0][1], pt_minus[1][0], pt_minus[1][1], cut_pt_list[k][free_axis])
                            y_v[1] = interpolate_scalar(pt_plus[0][0], pt_plus[0][1], pt_plus[1][0], pt_plus[1][1], cut_pt_list[k][free_axis])

                        pts = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
                        pts[0][acut[l]] = pts[1][acut[l]] = pts[2][acut[l]] = pts[3][acut[l]] = level[l]
                        pts[0][acut[l^1]] = pts[1][acut[l^1]] = x_inter[0]
                        pts[2][acut[l^1]] = pts[3][acut[l^1]] = x_inter[1]
                        pts[0][free_axis] = y_h[0]
                        pts[1][free_axis] = pts[2][free_axis] = cut_pt_list[k][free_axis]
                        pts[3][free_axis] = y_h[1]
                        
                        if (y_v[0] >= min(x_minus, x_plus) - DELTA and y_v[0] <= max(x_minus, x_plus) + DELTA):
                            pts[1][acut[l^1]] = y_v[0]
                            pts.pop(0)
                            idx_h[0] = idx_v[0]
                            idx_h[1] = idx_v[1]
#                        elif (y_v[1] >= min(x_minus, x_plus) - DELTA and y_v[1] <= max(x_minus, x_plus) + DELTA):
                        if (y_v[1] >= min(x_minus, x_plus) - DELTA and y_v[1] <= max(x_minus, x_plus) + DELTA):
                            pts[2][acut[l^1]] = y_v[1]
#                            pts.pop(3)
                            pts.pop(-1)
                            idx_h[2] = idx_v[2]
                            idx_h[3] = idx_v[3]

                        vt_insert = idx_h[0]+1
                        if idx_h[3] > idx_h[0]:
                            cnt = idx_h[3] - idx_h[0] - 1
                            for _ in range(cnt): vl[l].pop(idx_h[1])
                            vt_idx_added[l].append((idx_h[0], len(pts)-cnt))
                        else:
                            if idx_h[1] != spline_st:
                                for _ in range(idx_h[1], spline_end+1): vl[l].pop(idx_h[1])
                            vt_idx_added[l].append((idx_h[0], len(pts)-spline_end+idx_h[0]))
                            for _ in range(spline_st, idx_h[3]): vl[l].pop(spline_st)
                            vt_idx_added[l].append((spline_st, spline_st-idx_h[3]))
                            vt_insert -= (idx_h[3] - spline_st)

                        for m in range(len(pts)): vl[l].insert(vt_insert+m, pts[m])

                   
# ---------------------------------------------------------------------------------------------------------------
# WRITE FILE AND CREATE CURVES
    
    for idir in range(ndir):

        if bpy.data.collections.get(nameslice[acut[idir]]):
            ccl = bpy.data.collections[nameslice[acut[idir]]]
            scl = bpy.data.collections[nameslice[acut[idir]]+'S']
            bpy.ops.object.select_all(action='DESELECT')
            for o in ccl.objects:
                o.select_set(True)
            for o in scl.objects:
                o.select_set(True)
            bpy.ops.object.delete(use_global = True)
        else:
            ccl = bpy.data.collections.new(nameslice[acut[idir]])
            scl = bpy.data.collections.new(nameslice[acut[idir]]+'S')
            ccl[nameslice[acut[idir]]] = 1
            scl[nameslice[acut[idir]]+'S'] = 1
            bpy.context.scene.collection.children.link(scl)
            bpy.context.scene.collection.children.link(ccl)
        bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children[nameslice[acut[idir]]]
                

        if not sepfile:
            if os.path.isdir(bpy.path.abspath(os.path.join(bpy.path.abspath(ofile), aob.name+nameaxis[acut[idir]]))):
                filename = os.path.join(bpy.path.abspath(ofile), aob.name+nameaxis[acut[idir]]+'.svg')
            else:
                filename = os.path.join(os.path.dirname(bpy.data.filepath), aob.name+nameaxis[acut[idir]]+'.svg') if not ofile else bpy.path.abspath(os.path.splitext(ofile)[0]+nameaxis[acut[idir]]+'.svg')
        else:
            if os.path.isdir(bpy.path.abspath(os.path.join(bpy.path.abspath(ofile), aob.name+nameaxis[acut[idir]]))):
                filenames = [os.path.join(bpy.path.abspath(ofile), aob.name+nameaxis[acut[idir]]+'{:02d}.svg'.format(i)) for i in range(len(vlenlist))]
            else:
                if not ofile:
                    filenames = [os.path.join(os.path.dirname(bpy.path.abspath(bpy.data.filepath)), aob.name+nameaxis[acut[idir]]+'{:02d}.svg'.format(i)) for i in range(len(vlenlist))]
                else:
                    filenames = [bpy.path.abspath(os.path.splitext(ofile)[0]+nameaxis[acut[idir]] + '{:02d}.svg'.format(i)) for i in range(len(vlenlist))]

        yrowpos = 0
        xmaxlast = 0
        ydiff, rysize  = 0, 0
        
        for vci, vclist in enumerate(vtlist[idir]):
            if sepfile or vci == 0:
                svgtext = ''

            xmax = max([vc[0] for vc in vclist if vc not in (-1, -2)])
            xmin = min([vc[0] for vc in vclist if vc not in (-1, -2)])
            ymax = max([vc[1] for vc in vclist if vc not in (-1, -2)])
            ymin = min([vc[1] for vc in vclist if vc not in (-1, -2)])
            cxsize = xmax - xmin + ct
            cysize = ymax - ymin + ct
            
            zmax = max([vc[2] for vc in vclist if vc not in (-1, -2)])
            zmin = min([vc[2] for vc in vclist if vc not in (-1, -2)])
            czsize = zmax - zmin + ct

            if (acut[idir] == 0):
                xmax = ymax
                xmin = ymin
                ymax = zmax
                ymin = zmin
                cxsize = cysize
                cysize = czsize
            elif (acut[idir] == 1):
                ymax = xmax
                ymin = xmin
                xmax = zmax
                xmin = zmin
                cysize = cxsize
                cxsize = czsize
            
            if (sepfile and svgpos == '0') or (sepfile and vci == 0 and svgpos == '1'):
                xdiff = -xmin + ct
                ydiff = -ymin + ct

            elif (sepfile and svgpos == '1') or not sepfile:            
                if f_scale * (xmaxlast + cxsize) <= mwidth:
                    xdiff = xmaxlast - xmin + ct
                    ydiff = yrowpos - ymin + ct
                
                    if rysize < cysize:
                        rysize = cysize
                
                    xmaxlast += cxsize
                                
                elif f_scale * cxsize > mwidth:
                    xdiff = -xmin + ct
                    ydiff = yrowpos - ymin + ct
                    yrowpos += cysize
                    if rysize < cysize:
                        rysize = cysize
                
                    xmaxlast = cxsize
                    rysize = cysize
            
                else:
                    yrowpos += rysize
                    xdiff = -xmin + ct
                    ydiff = yrowpos - ymin + ct
                    xmaxlast = cxsize
                    rysize = cysize
        
            elif sepfile and svgpos == '2':
                xdiff = mwidth/(2 * f_scale) - (0.5 * cxsize) - xmin
                ydiff = mheight/(2 * f_scale) - (0.5 * cysize) - ymin

            elif sepfile and svgpos == '3':
                xdiff = 0
                ydiff = 0
                   
            if (acut[idir] == 0):
                v2p = [[vclist[0][1], vclist[0][2]], [vclist[1][1], vclist[1][2]]]
            elif (acut[idir] == 1):
                v2p = [[vclist[0][2], vclist[0][0]], [vclist[1][2], vclist[1][0]]]
            else:
                v2p = [[vclist[0][0], vclist[0][1]], [vclist[1][0], vclist[1][1]]]
            points = "{:.4f},{:.4f} {:.4f},{:.4f} ".format(scale*(xdiff+v2p[0][0]), scale*(ydiff+v2p[0][1]), scale*(xdiff+v2p[1][0]), scale*(ydiff+v2p[1][1]))
            svgtext += '<g>\n'

# build curve
            curveData = bpy.data.curves.new('Slices'+nameaxis[acut[idir]]+str(vci+1), type='CURVE')
            curveData.dimensions = '3D'
            polyline = curveData.splines.new('POLY')

            polyline.points.add(1)
            polyline.points[0].co = (vclist[0][0], vclist[0][1], vclist[0][2], 1)
            polyline.points[1].co = (vclist[1][0], vclist[1][1], vclist[1][2], 1)
# build curve
            
            for vco in vclist[2:]:
                if vco in (-1, -2):
                    polyend = 'gon' if vco == -1 else 'line'
                    svgtext += '<poly{0} points="{1}" style="fill:none;stroke:rgb({2[0]},{2[1]},{2[2]});stroke-width:{3}" />\n'.format(polyend, points, [int(255 * lc) for lc in lcol], lthick)
                    points = '' 
# build curve
                    polyline.use_cyclic_u = True
# build curve
                else:
                    if (acut[idir] == 0):
                        v2p = [vco[1], vco[2] ]
                    elif (acut[idir] == 1):
                        v2p = [vco[2], vco[0] ]
                    else:
                        v2p = [vco[0], vco[1] ]
# build curve
                    if len(points) == 0:
                        polyline = curveData.splines.new('POLY')
                    else:
                        polyline.points.add(1)
                    iend = len(polyline.points) - 1
                    polyline.points[iend].co = (vco[0], vco[1], vco[2], 1)
# build curve
                    points += "{:.4f},{:.4f} ".format(scale*(xdiff+v2p[0]), scale*(ydiff+v2p[1]))
             
            if points:
                svgtext += '<polygon points="{0}" style="fill:none;stroke:rgb({1[0]},{1[1]},{1[2]});stroke-width:{2}" />\n'.format(points, [int(255 * lc) for lc in lcol], lthick)
# build curve
                polyline.use_cyclic_u = True
# build curve
             
            svgtext += '</g>\n'          

# build curve422
            curveOB = bpy.data.objects.new('Slices'+nameaxis[acut[idir]]+str(vci+1), curveData)
            bpy.context.collection.objects.link(curveOB)
# build curve

            if sepfile:
                svgtext += '</svg>\n' 
                
                with open(filenames[vci], 'w') as svgfile:
                    svgfile.write('<?xml version="1.0"?>\n<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n\
                    <svg xmlns="http://www.w3.org/2000/svg" version="1.1"\n    width="{0}"\n    height="{1}"\n    viewbox="0 0 {0} {1}">\n\
                    <desc>Laser SVG Slices from Object: Sphere_net. Exported from Blender3D with the Laser Slicer Script</desc>\n\n'.format(mwidth*mm2pi, mheight*mm2pi))
                    
                    svgfile.write(svgtext)
            
        if not sepfile:
            
            with open(filename, 'w') as svgfile:
                svgfile.write('<?xml version="1.0"?>\n<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n\
                    <svg xmlns="http://www.w3.org/2000/svg" version="1.1"\n    width="{0}"\n    height="{1}"\n    viewbox="0 0 {0} {1}">\n\
                    <desc>Laser SVG Slices from Object: Sphere_net. Exported from Blender3D with the Laser Slicer Script</desc>\n\n'.format(mwidth*mm2pi, mheight*mm2pi))
                
                svgfile.write(svgtext)
                svgfile.write("</svg>\n")
                
        svgfile.close()

# add a mesh from curve and solidify with thickness            
    for idir in range(ndir):
        ccl = bpy.data.collections[nameslice[acut[idir]]]
        scl = bpy.data.collections[nameslice[acut[idir]]+'S']
        for o in ccl.objects:
            if (o.type == 'CURVE'):
                bpy.ops.object.select_all(action='DESELECT')
                o.select_set(True)
                bpy.context.view_layer.objects.active = o
                bpy.ops.object.convert(target='MESH', keep_original=True)
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.fill()
                bpy.ops.object.mode_set(mode='OBJECT')
                bpy.ops.object.modifier_add(type='SOLIDIFY')
                bpy.context.object.modifiers["Solidify"].offset = 0
                bpy.context.object.modifiers["Solidify"].thickness = lt
                scl.objects.link(bpy.context.active_object)
                ccl.objects.unlink(bpy.context.active_object)
                s = bpy.context.active_object.name
                i = s.find('.')
                bpy.context.active_object.name = (s[:6]+'S'+s[6:], s[:6]+'S'+s[6:i])[i > -1]

# restore the active object
    bpy.ops.object.select_all(action='DESELECT')
    aob.select_set(True)
    bpy.context.view_layer.objects.active = aob

class OBJECT_OT_X_Slicer(bpy.types.Operator):
    bl_label = "X-Slicer"
    bl_idname = "object.x_slicer"

    def execute(self, context):
        slicer(context.scene.slicer_settings)
        return {'FINISHED'}

class XSlicerPanel(bpy.types.Panel):
    bl_idname = "OBJECT_PT_X_Slicer_Panel"
    bl_label = "X-Slicer Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_context = "objectmode"
    bl_category = "X-Slicer"

    def draw(self, context):
        scene = context.scene
        layout = self.layout
        ss = scene.slicer_settings
        d = int(ss.x_slicer_cut_plane)
        d_2 = int(ss.x_slicer_cut_plane_2)
        row = layout.row()
        row.label(text = "Material dimensions:")
        newrow(layout, "Thickness (mm):", scene.slicer_settings, 'x_slicer_material_thick', True)
        newrow(layout, "Width (mm):", scene.slicer_settings, 'x_slicer_material_width', True)
        newrow(layout, "Height (mm):", scene.slicer_settings, 'x_slicer_material_height', True)
        layout.row().separator()
        row = layout.row()
        row.label(text = "Slice settings:")
        newrow(layout, "No. of planes:", scene.slicer_settings, 'x_slicer_cut_ndir', True)
        
        newrow(layout, "Axis:", scene.slicer_settings, 'x_slicer_cut_plane', True)
        
        sameside = False
        mind = 0
        maxd = 0
        if context.active_object and context.active_object.select_get() and context.active_object.type == 'MESH' and context.active_object.data.polygons:
            omw = context.active_object.matrix_world
            vv = [omw @ Vector((v[0], v[1], v[2])) for v in context.active_object.bound_box]
            maxd = max (v[d] for v in vv)
            mind = min (v[d] for v in vv)
            sameside = (False, True)[maxd * mind >= 0]
            
        newrow(layout, "No. of sides:", scene.slicer_settings, 'x_slicer_cut_sides', not sameside)
        newrow(layout, "Center cut:", scene.slicer_settings, 'x_slicer_cut_center', ss.x_slicer_cut_sides == 2 and not sameside)
        newrow(layout, "1st cut (mm):", scene.slicer_settings, 'x_slicer_cut_first_loc', not ss.x_slicer_cut_center or ss.x_slicer_cut_sides == 1 or sameside)
        newrow(layout, "Spacing (mm):", scene.slicer_settings, 'x_slicer_cut_spacing', True)

        if ss.x_slicer_cut_ndir == 2:
            layout.row().separator()
            newrow(layout, "Axis:", scene.slicer_settings, 'x_slicer_cut_plane_2', True)
        
            sameside_2 = False
            mind_2 = 0
            maxd_2 = 0
            if context.active_object and context.active_object.select_get() and context.active_object.type == 'MESH' and context.active_object.data.polygons:
                maxd_2 = max (v[d_2] for v in vv)
                mind_2 = min (v[d_2] for v in vv)
                sameside_2 = (False, True)[maxd_2 * mind_2 >= 0]
            
            newrow(layout, "No. of sides:", scene.slicer_settings, 'x_slicer_cut_sides_2', not sameside_2)
            newrow(layout, "Center cut:", scene.slicer_settings, 'x_slicer_cut_center_2', ss.x_slicer_cut_sides_2 == 2 and not sameside_2)
            newrow(layout, "1st cut (mm):", scene.slicer_settings, 'x_slicer_cut_first_loc_2', not ss.x_slicer_cut_center_2 or ss.x_slicer_cut_sides_2 == 1 or sameside_2)
            newrow(layout, "Spacing (mm):", scene.slicer_settings, 'x_slicer_cut_spacing_2', True)

        layout.row().separator()
        row = layout.row()
        row.label(text = "Export settings:")
        newrow(layout, "DPI:", scene.slicer_settings, 'x_slicer_dpi', True)
        newrow(layout, "Line colour:", scene.slicer_settings, 'x_slicer_cut_colour', True)
        newrow(layout, "Thickness (pixels):", scene.slicer_settings, 'x_slicer_cut_line', True)
        newrow(layout, "Separate files:", scene.slicer_settings, 'x_slicer_separate_files', True)

        if scene.slicer_settings.x_slicer_separate_files:
            newrow(layout, "Cut position:", scene.slicer_settings, 'x_slicer_svg_position', True)

        newrow(layout, "Cut spacing (mm):", scene.slicer_settings, 'x_slicer_cut_thickness', True)
        newrow(layout, "Export file(s):", scene.slicer_settings, 'x_slicer_ofile', True)

        if context.active_object and context.active_object.select_get() and context.active_object.type == 'MESH' and context.active_object.data.polygons:
            row = layout.row()
            
            no_slices = 0
            if (ss.x_slicer_cut_sides == 1) or (sameside == True):
                if (bpy.context.active_object.dimensions[d] * 1000 * bpy.context.scene.unit_settings.scale_length - ss.x_slicer_cut_first_loc > ss.x_slicer_material_thick):
                    no_slices = (bpy.context.active_object.dimensions[d] * 1000 * bpy.context.scene.unit_settings.scale_length - ss.x_slicer_cut_first_loc + ss.x_slicer_cut_spacing + 0.5*ss.x_slicer_material_thick)/(ss.x_slicer_material_thick + ss.x_slicer_cut_spacing)
                    no_slices = math.floor (no_slices)
            else:
                if (ss.x_slicer_cut_center == True):
                    if (maxd * 1000 * bpy.context.scene.unit_settings.scale_length - 0.5 * ss.x_slicer_material_thick > ss.x_slicer_material_thick + ss.x_slicer_cut_spacing):
                        no_slices_p = (maxd * 1000 * bpy.context.scene.unit_settings.scale_length - 0.5 * ss.x_slicer_material_thick)/(ss.x_slicer_material_thick + ss.x_slicer_cut_spacing)
                    if (-mind * 1000 * bpy.context.scene.unit_settings.scale_length - 0.5 * ss.x_slicer_material_thick > ss.x_slicer_material_thick + ss.x_slicer_cut_spacing):
                        no_slices_n = (-mind * 1000 * bpy.context.scene.unit_settings.scale_length - 0.5 * ss.x_slicer_material_thick)/(ss.x_slicer_material_thick + ss.x_slicer_cut_spacing)
                    no_slices = math.floor(no_slices_p) + math.floor(no_slices_n) + 1
                else:
                    no_slices_p = 0
                    if (maxd * 1000 * bpy.context.scene.unit_settings.scale_length - ss.x_slicer_cut_first_loc > ss.x_slicer_material_thick):
                        no_slices_p = (maxd * 1000 * bpy.context.scene.unit_settings.scale_length - ss.x_slicer_cut_first_loc + ss.x_slicer_cut_spacing + 0.5*ss.x_slicer_material_thick)/(ss.x_slicer_material_thick + ss.x_slicer_cut_spacing)
                    no_slices_n = 0
                    if (-mind * 1000 * bpy.context.scene.unit_settings.scale_length - ss.x_slicer_cut_first_loc > ss.x_slicer_material_thick):
                        no_slices_n = (-mind * 1000 * bpy.context.scene.unit_settings.scale_length - ss.x_slicer_cut_first_loc + ss.x_slicer_cut_spacing + 0.5*ss.x_slicer_material_thick)/(ss.x_slicer_material_thick + ss.x_slicer_cut_spacing)
                    no_slices = math.floor(no_slices_p) + math.floor(no_slices_n)

            row.label(text = 'No. of slices : {:.0f}'.format(no_slices))
            
            if ss.x_slicer_cut_ndir == 2:
                no_slices = 0
                if (ss.x_slicer_cut_sides_2 == 1) or (sameside_2 == True):
                    if (bpy.context.active_object.dimensions[d] * 1000 * bpy.context.scene.unit_settings.scale_length - ss.x_slicer_cut_first_loc_2 > ss.x_slicer_material_thick):
                        no_slices = (bpy.context.active_object.dimensions[d] * 1000 * bpy.context.scene.unit_settings.scale_length - ss.x_slicer_cut_first_loc_2 + ss.x_slicer_cut_spacing_2 + 0.5*ss.x_slicer_material_thick)/(ss.x_slicer_material_thick + ss.x_slicer_cut_spacing_2)
                        no_slices = math.floor (no_slices)
                else:
                    if (ss.x_slicer_cut_center_2 == True):
                        if (maxd_2 * 1000 * bpy.context.scene.unit_settings.scale_length - 0.5 * ss.x_slicer_material_thick > ss.x_slicer_material_thick + ss.x_slicer_cut_spacing_2):
                            no_slices_p = (maxd * 1000 * bpy.context.scene.unit_settings.scale_length - 0.5 * ss.x_slicer_material_thick)/(ss.x_slicer_material_thick + ss.x_slicer_cut_spacing_2)
                        if (-mind_2 * 1000 * bpy.context.scene.unit_settings.scale_length - 0.5 * ss.x_slicer_material_thick > ss.x_slicer_material_thick + ss.x_slicer_cut_spacing_2):
                            no_slices_n = (-mind * 1000 * bpy.context.scene.unit_settings.scale_length - 0.5 * ss.x_slicer_material_thick)/(ss.x_slicer_material_thick + ss.x_slicer_cut_spacing_2)
                        no_slices = math.floor(no_slices_p) + math.floor(no_slices_n) + 1
                    else:
                        no_slices_p = 0
                        if (maxd_2 * 1000 * bpy.context.scene.unit_settings.scale_length - ss.x_slicer_cut_first_loc_2 > ss.x_slicer_material_thick):
                            no_slices_p = (maxd_2 * 1000 * bpy.context.scene.unit_settings.scale_length - ss.x_slicer_cut_first_loc_2 + ss.x_slicer_cut_spacing_2 + 0.5*ss.x_slicer_material_thick)/(ss.x_slicer_material_thick + ss.x_slicer_cut_spacing_2)
                        no_slices_n = 0
                        if (-mind_2 * 1000 * bpy.context.scene.unit_settings.scale_length - ss.x_slicer_cut_first_loc_2 > ss.x_slicer_material_thick):
                            no_slices_n = (-mind_2 * 1000 * bpy.context.scene.unit_settings.scale_length - ss.x_slicer_cut_first_loc_2 + ss.x_slicer_cut_spacing_2 + 0.5*ss.x_slicer_material_thick)/(ss.x_slicer_material_thick + ss.x_slicer_cut_spacing_2)
                        no_slices = math.floor(no_slices_p) + math.floor(no_slices_n)

                row = layout.row()
                row.label(text = 'No. of slices (2): {:.0f}'.format(no_slices))

            if bpy.data.filepath or scene.slicer_settings.x_slicer_ofile:
                split = layout.split()
                col = split.column()
                col.operator("object.x_slicer", text="Slice the object")

def update_cut_plane(self, context):
    ss = context.scene.slicer_settings
    if int(ss.x_slicer_cut_plane) == int(ss.x_slicer_cut_plane_2):
       ss.x_slicer_cut_plane = str((int(ss.x_slicer_cut_plane) + 1) % 3)

def update_cut_plane_2(self, context):
    ss = context.scene.slicer_settings
    if int(ss.x_slicer_cut_plane_2) == int(ss.x_slicer_cut_plane):
       ss.x_slicer_cut_plane_2 = str((int(ss.x_slicer_cut_plane_2) + 1) % 3)

class Slicer_Settings(bpy.types.PropertyGroup):   
    x_slicer_material_thick: FloatProperty(
         name="", description="Thickness of the cutting material in mm",
             min=0.1, max=50, default=2)
    x_slicer_material_width: FloatProperty(
         name="", description="Width of the cutting material in mm",
             min=1, max=5000, default=450)
    x_slicer_material_height: FloatProperty(
         name="", description="Height of the cutting material in mm",
             min=1, max=5000, default=450)
    x_slicer_cut_ndir: IntProperty(
         name="", description="no. of cutting planes",
             min=1, max=2, default=1)

    x_slicer_cut_plane: EnumProperty(items = [('0', 'X', 'along X axis'), ('1', 'Y', 'along Y axis'), ('2', 'Z', 'along Z axis')], name = "", description = "cut direction", default = '2', update=update_cut_plane_2)
    x_slicer_cut_sides: IntProperty(
         name="", description="1 or 2 directions of cut about cutting plane",
             min=1, max=2, default=1)
    x_slicer_cut_center: BoolProperty(
         name="", description="center in 2 direction cut",
             default=False)
    x_slicer_cut_first_loc: FloatProperty(
         name="", description="First cut from plane in mm",
             min=0, max=50, default=0)
    x_slicer_cut_spacing: FloatProperty(
         name="", description="Spacing of the cutting material in mm",
             min=0, max=50, default=0)

    x_slicer_cut_plane_2: EnumProperty(items = [('0', 'X', 'along X axis'), ('1', 'Y', 'along Y axis'), ('2', 'Z', 'along Z axis')], name = "", description = "cut direction", default = '2', update=update_cut_plane_2)
    x_slicer_cut_sides_2: IntProperty(
         name="", description="1 or 2 directions of cut about cutting plane",
             min=1, max=2, default=1)
    x_slicer_cut_center_2: BoolProperty(
         name="", description="center in 2 direction cut",
             default=False)
    x_slicer_cut_first_loc_2: FloatProperty(
         name="", description="First cut from plane in mm",
             min=0, max=50, default=0)
    x_slicer_cut_spacing_2: FloatProperty(
         name="", description="Spacing of the cutting material in mm",
             min=0, max=50, default=0)

    x_slicer_dpi: IntProperty(
         name="", description="DPI of the laser cutter computer",
             min=50, max=500, default=96)
    x_slicer_separate_files: BoolProperty(name = "", description = "Write out seperate SVG files", default = 0)
    x_slicer_svg_position: EnumProperty(items = [('0', 'Top left', 'Keep top  left position'), ('1', 'Staggered', 'Staggered position'), ('2', 'Centre', 'Apply centre position'), ('3', 'Original', 'No change in position')], name = "", description = "Control the position of the SVG slice", default = '0')
    x_slicer_cut_thickness: FloatProperty(
         name="", description="Expected thickness of the laser cut (mm)",
             min=0, max=5, default=1)
    x_slicer_ofile: StringProperty(name="", description="Location of the exported file", default="", subtype="FILE_PATH")
    x_slicer_cut_colour: FloatVectorProperty(size = 3, name = "", attr = "Lini colour", default = [1.0, 0.0, 0.0], subtype ='COLOR', min = 0, max = 1)
    x_slicer_cut_line: FloatProperty(name="", description="Thickness of the svg line (pixels)", min=0, max=5, default=1)

classes = (XSlicerPanel, OBJECT_OT_X_Slicer, Slicer_Settings)

def register():
    for cl in classes:
        bpy.utils.register_class(cl)

    bpy.types.Scene.slicer_settings = bpy.props.PointerProperty(type=Slicer_Settings)

def unregister():
    bpy.types.Scene.slicer_settings
    
    for cl in classes:
        bpy.utils.unregister_class(cl)

