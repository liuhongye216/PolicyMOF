from __future__ import print_function
import sys, os
# Directory containing the external TOBACCO modules required below.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BASE_DIR = os.environ.get("MOF_TOBACCO_BASE_DIR", os.path.join(_REPO_ROOT, "tobacco_workdir"))
sys.path.insert(0, BASE_DIR)
from ciftemplate2graph import ct2g
from vertex_edge_assign import vertex_assign, assign_node_vecs2edges
from cycle_cocyle import cycle_cocyle, Bstar_alpha
from bbcif_properties import cncalc, bbelems
from SBU_geometry import SBU_coords
from scale import scale
from scaled_embedding2coords import omega2coords
from place_bbs import scaled_node_and_edge_vectors, place_nodes, place_edges
from remove_net_charge import fix_charges
from remove_dummy_atoms import remove_Fr
from adjust_edges import adjust_edges
from write_cifs import write_check_cif, write_cif, bond_connected_components, distance_search_bond, fix_bond_sym, merge_catenated_cifs
from scale_animation import scaling_callback_animation, write_scaling_callback_animation, animate_objective_minimization
import time
import configuration
import os
import re
import numpy as np
import itertools
import time
import glob
import multiprocessing
from random import choice

####### Global options #######
IGNORE_ALL_ERRORS = configuration.IGNORE_ALL_ERRORS
PRINT = configuration.PRINT
CONNECTION_SITE_BOND_LENGTH = configuration.CONNECTION_SITE_BOND_LENGTH
WRITE_CHECK_FILES = configuration.WRITE_CHECK_FILES
WRITE_CIF = configuration.WRITE_CIF
ALL_NODE_COMBINATIONS = configuration.ALL_NODE_COMBINATIONS
USER_SPECIFIED_NODE_ASSIGNMENT = configuration.USER_SPECIFIED_NODE_ASSIGNMENT
COMBINATORIAL_EDGE_ASSIGNMENT = configuration.COMBINATORIAL_EDGE_ASSIGNMENT
CHARGES = configuration.CHARGES
SCALING_ITERATIONS = configuration.SCALING_ITERATIONS
SYMMETRY_TOL = configuration.SYMMETRY_TOL
BOND_TOL = configuration.BOND_TOL
ORIENTATION_DEPENDENT_NODES = configuration.ORIENTATION_DEPENDENT_NODES
PLACE_EDGES_BETWEEN_CONNECTION_POINTS = configuration.PLACE_EDGES_BETWEEN_CONNECTION_POINTS
RECORD_CALLBACK = configuration.RECORD_CALLBACK
OUTPUT_SCALING_DATA = configuration.OUTPUT_SCALING_DATA
FIX_UC = configuration.FIX_UC
MIN_CELL_LENGTH = configuration.MIN_CELL_LENGTH
OPT_METHOD = configuration.OPT_METHOD
PRE_SCALE = configuration.PRE_SCALE
SINGLE_METAL_MOFS_ONLY = configuration.SINGLE_METAL_MOFS_ONLY
MOFS_ONLY = configuration.MOFS_ONLY
MERGE_CATENATED_NETS = configuration.MERGE_CATENATED_NETS
RUN_PARALLEL = configuration.RUN_PARALLEL
REMOVE_DUMMY_ATOMS = configuration.REMOVE_DUMMY_ATOMS
FOLDER_TIMEOUT = 180
####### Global options #######

pi = np.pi

vname_dict = {'V':1,'Er':2,'Ti':3,'Ce':4,'S':5,
			'H':6,'He':7,'Li':8,'Be':9,'B':10,
			'C':11,'N':12,'O':13,'F':14,'Ne':15,
			'Na':16,'Mg':17,'Al':18,'Si':19,'P':20 ,
			'Cl':21,'Ar':22,'K':23,'Ca':24,'Sc':24,
			'Cr':26,'Mn':27,'Fe':28,'Co':29,'Ni':30}

metal_elements = ['Ac','Ag','Al','Am','Au','Ba','Be','Bi',
				'Bk','Ca','Cd','Ce','Cf','Cm','Co','Cr',
				'Cs','Cu','Dy','Er','Es','Eu','Fe','Fm',
				'Ga','Gd','Hf','Hg','Ho','In','Ir',
				'K','La','Li','Lr','Lu','Md','Mg','Mn',
				'Mo','Na','Nb','Nd','Ni','No','Np','Os',
				'Pa','Pb','Pd','Pm','Pr','Pt','Pu','Ra',
				'Rb','Re','Rh','Ru','Sc','Sm','Sn','Sr',
				'Ta','Tb','Tc','Th','Ti','Tl','Tm','U',
				'V','W','Y','Yb','Zn','Zr']

def parse_edge_folder_meta(folder_path):
	"""
	解析edges文件夹下子文件夹中的meta.txt
	返回: (topology_name, metal_node_list, cif_files_list)
	注意：metal_node_list现在是一个列表，支持逗号分隔的多个node
	"""
	meta_path = os.path.join(folder_path, 'meta.txt')

	if not os.path.exists(meta_path):
		return None, None, []
	
	try:
		with open(meta_path, 'r', encoding='utf-8') as f:
			lines = f.readlines()
		
		if len(lines) < 3:
			return None, None, []
		
		# 第二行: 拓扑: MOFid-v1.pcu.cat0
		topology_line = lines[1].strip()
		if '拓扑:' in topology_line or 'topology' in topology_line.lower():
			parts = topology_line.split(':')[-1].strip().split('.')
			if len(parts) >= 2:
				topology_name = parts[1]  # 提取pcu
			else:
				topology_name = None
		else:
			topology_name = None
		
		metal_line = lines[2].strip()
		metal_node_list = None

		if '金属节点:' in metal_line or 'metal' in metal_line.lower():
			metal_str = metal_line.split(':')[-1].strip()
			# 按逗号分割，去除空格
			metal_node_list = [node.strip() for node in metal_str.split(',') if node.strip()]
			
			if len(metal_node_list) == 0:
				metal_node_list = None

		# 获取该文件夹下所有.cif文件
		cif_files = [f for f in os.listdir(folder_path) if f.endswith('.cif')]

		return topology_name, metal_node_list, cif_files
		
	except Exception as e:
		print(f"Error parsing meta.txt in {folder_path}: {e}")
		return None, None, []

def get_edge_folder_assignments():
	"""
	扫描edges目录下所有子文件夹，返回edge assignment列表
	每个assignment是一个字典: {
		'folder': 文件夹名,
		'topology': 拓扑名,
		'metal_node_list': 金属节点列表,
		'cif_files': [cif文件列表],
		'template_file': 对应的template文件,
		'has_issues': 是否有问题,
		'issues': 问题列表
	}
	"""
	edge_assignments = []
	failed_assignments = []
	edges_dir = 'edges'
	
	if not os.path.exists(edges_dir):
		print(f"Error: {edges_dir} directory not found!")
		return edge_assignments, failed_assignments
	
	for folder_name in os.listdir(edges_dir):
		folder_path = os.path.join(edges_dir, folder_name)
		
		if not os.path.isdir(folder_path):
			continue
		
		topology, metal_node_list, cif_files = parse_edge_folder_meta(folder_path)
		
		issues = []
		has_issues = False
		
		if not topology:
			issues.append('No topology found in meta.txt')
			has_issues = True
		
		if not metal_node_list or len(metal_node_list) == 0:
			issues.append('No metal_node_list found in meta.txt')
			has_issues = True
		
		if not cif_files:
			issues.append('No CIF files found in folder')
			has_issues = True
		
		# 查找对应的template文件
		template_file = None
		if topology:
			templates_dir = 'templates'
			if os.path.exists(templates_dir):
				target_template = f"{topology}.cif"
				if target_template in os.listdir(templates_dir):
					template_file = target_template
				else:
					issues.append(f'No template file found for topology: {topology}')
					has_issues = True
		
		assignment = {
			'folder': folder_name,
			'topology': topology,
			'metal_node_list': metal_node_list,  # 改为metal_node_list
			'cif_files': cif_files,
			'template_file': template_file,
			'folder_path': folder_path,
			'has_issues': has_issues,
			'issues': issues
		}
				
		if has_issues:
			failed_assignments.append(assignment)
		else:
			edge_assignments.append(assignment)
		
		print(f"Scanned: {folder_name} - Issues: {has_issues}")
		if has_issues:
			print(f"  Problems: {', '.join(issues)}")
	
	return edge_assignments, failed_assignments

def find_matching_nodes(metal_node_list):
	"""
	根据金属节点列表在nodes目录中找到匹配的节点文件列表
	
	Args:
		metal_node_list: node字符串列表，如 ['[Cu][Cu]', '[Zn][Zn]']
	
	Returns:
		matching_nodes: 匹配的node文件列表，如 ['[Cu][Cu].cif', '[Zn][Zn].cif']
		如果任何一个node没找到，返回None
	"""
	nodes_dir = 'nodes'
	if not os.path.exists(nodes_dir):
		return None
	
	if not metal_node_list or len(metal_node_list) == 0:
		return None
	
	matching_nodes = []
	
	for metal_node_str in metal_node_list:
		# 清理metal_node_str，移除空格
		metal_node_clean = metal_node_str.strip().replace(' ', '')
		
		# 在nodes目录中查找精确匹配的文件（忽略 .cif 扩展名、忽略空格）
		found = False
		for node_file in os.listdir(nodes_dir):
			if not node_file.endswith('.cif'):
				continue
			
			base = node_file[:-4]  # 去掉 .cif
			if base.replace(' ', '') == metal_node_clean:
				matching_nodes.append(node_file)
				found = True
				break
		
		if not found:
			print(f"  ✗ No matching node file found for: {metal_node_str}")
			return None
	
	return matching_nodes

def count_X_atoms_in_node(node_cif):
	"""
	计算node CIF文件中X原子的数量
	
	Args:
		node_cif: node文件名，如 '[Cu][Cu].cif'
	
	Returns:
		x_count: X原子数量
	"""
	from bbcif_properties import cncalc
	
	try:
		x_count = cncalc(node_cif, 'nodes')
		return x_count
	except Exception as e:
		print(f"    Warning: Could not count X atoms in {node_cif}, assuming 0")
		return 0

def create_vertex_assignments(TG, TVT, matching_nodes, folder_name):
	"""
	根据拓扑顶点类型和匹配的node列表创建vertex assignments
	
	策略：
	1. 如果拓扑只需要1种node，且列表也只有1个元素 → 所有顶点用同一个node
	2. 如果拓扑需要n种node，且列表也有n个元素 → 按CN和X数量匹配分配
	3. 如果拓扑需要2种node，但列表只有1个元素 → 复用该node（带警告）
	4. 其他情况 → 跳过
	
	Args:
		TG: 拓扑图
		TVT: 拓扑顶点类型集合，如 {(4, 'V'), (3, 'Er')}
		matching_nodes: 匹配的node文件列表
		folder_name: 文件夹名（用于日志）
	
	Returns:
		vas: vertex assignment列表
		skip_reason: 如果需要跳过，返回原因；否则返回None
	"""
	
	num_vertex_types = len(TVT)
	num_nodes = len(matching_nodes)
	
	print(f"  Topology requires {num_vertex_types} vertex type(s)")
	print(f"  Meta.txt provides {num_nodes} node(s): {matching_nodes}")
	
	# 情况1: 拓扑需要1种顶点，列表有1个node
	if num_vertex_types == 1 and num_nodes == 1:
		print(f"  ✓ Case 1: Single vertex type + single node → all vertices use {matching_nodes[0]}")
		vas = []
		va = []
		for node in TG.nodes():
			va.append((node, matching_nodes[0]))
		vas = [va]
		return vas, None
	
	# 情况2: 拓扑需要n种顶点，列表有n个node
	elif num_vertex_types == num_nodes and num_vertex_types > 1:
		print(f"  ✓ Case 2: {num_vertex_types} vertex types + {num_nodes} nodes → matching by CN and X count")
		
		# 1️⃣ 按照TVT的CN从大到小排序
		TVT_sorted = sorted(TVT, key=lambda x: x[0], reverse=True)
		print(f"    Topology vertex types (sorted by CN):")
		for cn, vtype in TVT_sorted:
			print(f"      - Type '{vtype}': CN={cn}")
		
		# 2️⃣ 计算每个node的X原子数量，并按X数量从大到小排序
		node_x_counts = []
		for node_file in matching_nodes:
			x_count = count_X_atoms_in_node(node_file)
			node_x_counts.append((node_file, x_count))
		
		# 按X数量降序排序
		node_x_counts_sorted = sorted(node_x_counts, key=lambda x: x[1], reverse=True)
		
		print(f"    Nodes (sorted by X atom count):")
		for node_file, x_count in node_x_counts_sorted:
			print(f"      - {node_file}: X atoms={x_count}")
		
		# 3️⃣ 检查CN和X数量是否匹配
		mismatch_warnings = []
		for i, ((cn, vtype), (node_file, x_count)) in enumerate(zip(TVT_sorted, node_x_counts_sorted)):
			if cn != x_count:
				warning = f"Type '{vtype}' (CN={cn}) ← {node_file} (X={x_count})"
				mismatch_warnings.append(warning)
		
		if mismatch_warnings:
			print(f"    ⚠ Warning: CN and X count mismatch detected:")
			for warning in mismatch_warnings:
				print(f"      - {warning}")
			print(f"    Proceeding anyway, but structure may be incorrect!")
		
		# 4️⃣ 创建映射: vertex type -> node file
		type_to_node = {}
		for i, (cn, vtype) in enumerate(TVT_sorted):
			if i < len(node_x_counts_sorted):
				node_file = node_x_counts_sorted[i][0]
				x_count = node_x_counts_sorted[i][1]
				type_to_node[vtype] = node_file
				print(f"    ✓ Assignment: Vertex type '{vtype}' (CN={cn}) → {node_file} (X={x_count})")
		
		# 5️⃣ 创建vertex assignment
		vas = []
		va = []
		for node_name, node_data in TG.nodes(data=True):
			vtype = node_data['type']
			if vtype in type_to_node:
				va.append((node_name, type_to_node[vtype]))
			else:
				# 这不应该发生，但作为安全措施
				skip_reason = f"Vertex type '{vtype}' not in type_to_node mapping"
				print(f"  ✗ ERROR: {skip_reason}")
				return None, skip_reason
		
		vas = [va]
		return vas, None
	
	# 情况3: 拓扑需要2种顶点，但只提供1个node → 复用（带警告）
	elif num_vertex_types == 2 and num_nodes == 1:
		print(f"  ⚠ Case 3: Topology needs 2 vertex types, but only 1 node provided → REUSING node for both types")
		
		# 获取node的X原子数量
		node_file = matching_nodes[0]
		x_count = count_X_atoms_in_node(node_file)
		print(f"    Node {node_file} has {x_count} X atoms")
		
		# 检查拓扑的CN要求
		TVT_sorted = sorted(TVT, key=lambda x: x[0], reverse=True)
		cns = [cn for cn, vtype in TVT_sorted]
		vtypes = [vtype for cn, vtype in TVT_sorted]
		
		print(f"    Topology vertex types:")
		for cn, vtype in TVT_sorted:
			print(f"      - Type '{vtype}': CN={cn}")
		
		# 检查CN是否都匹配
		cn_mismatches = []
		for cn, vtype in TVT_sorted:
			if cn != x_count:
				cn_mismatches.append((vtype, cn, x_count))
		
		if cn_mismatches:
			print(f"    ⚠⚠ WARNING: CN mismatch detected!")
			for vtype, required_cn, actual_x in cn_mismatches:
				print(f"      - Type '{vtype}' requires CN={required_cn}, but node has X={actual_x}")
			print(f"    ⚠⚠ The resulting structure may be INCORRECT!")
			print(f"    ⚠⚠ Consider providing 2 different node files matching the topology requirements")
		else:
			print(f"    ✓ Both vertex types require CN={x_count}, which matches the node's X count")
			print(f"    Structure should be correct (though using same node for different topological positions)")
		
		# 创建映射: 所有顶点类型都映射到同一个node
		type_to_node = {}
		for cn, vtype in TVT_sorted:
			type_to_node[vtype] = node_file
			print(f"    ✓ Assignment: Vertex type '{vtype}' (CN={cn}) → {node_file} (X={x_count}) [REUSED]")
		
		# 创建vertex assignment
		vas = []
		va = []
		for node_name, node_data in TG.nodes(data=True):
			vtype = node_data['type']
			if vtype in type_to_node:
				va.append((node_name, type_to_node[vtype]))
			else:
				skip_reason = f"Vertex type '{vtype}' not in type_to_node mapping"
				print(f"  ✗ ERROR: {skip_reason}")
				return None, skip_reason
		
		vas = [va]
		
		# 记录警告到文件
		with open('node_reuse_warnings.txt', 'a', encoding='utf-8') as f:
			f.write(f"\nFolder: {folder_name}\n")
			f.write(f"  Topology: {num_vertex_types} vertex types (CNs: {cns})\n")
			f.write(f"  Provided: 1 node ({node_file}, X={x_count})\n")
			f.write(f"  Action: REUSED node for both vertex types\n")
			if cn_mismatches:
				f.write(f"  ⚠ CN MISMATCH DETECTED - Structure may be incorrect!\n")
			f.write('-' * 80 + '\n')
		
		return vas, None
	
	# 情况4: 其他数量不匹配的情况
	else:
		skip_reason = f"Mismatch: topology needs {num_vertex_types} vertex types, but meta.txt provides {num_nodes} nodes"
		print(f"  ✗ SKIP: {skip_reason}")
		
		# 给出建议
		if num_nodes == 1 and num_vertex_types > 2:
			print(f"    💡 Suggestion: This topology requires {num_vertex_types} different vertex types.")
			print(f"    💡 Please provide {num_vertex_types} different node files, or consider if node reuse is appropriate.")
		elif num_nodes > 1 and num_nodes < num_vertex_types:
			print(f"    💡 Suggestion: Topology needs {num_vertex_types} nodes, but you provided {num_nodes}.")
			print(f"    💡 Please provide {num_vertex_types - num_nodes} more node file(s).")
		elif num_nodes > num_vertex_types:
			print(f"    💡 Suggestion: You provided {num_nodes} nodes, but topology only needs {num_vertex_types}.")
			print(f"    💡 The system doesn't know which nodes to use. Please provide exactly {num_vertex_types} node(s).")
		
		return None, skip_reason

def generate_short_cifname(template_name, node_name, edge_folder, edge_count):
	"""
	生成简短的cif文件名，避免超长和重名
	使用文件夹名的hash值确保唯一性
	"""
	import hashlib
	
	# 提取关键部分
	topo = template_name.replace('.cif', '').split('_')[0][:10]
	node_short = node_name.replace('.cif', '').split('_')[0][:15]
	
	# 使用完整文件夹名生成短hash（8位）确保唯一性
	folder_hash = hashlib.md5(edge_folder.encode()).hexdigest()[:8]
	
	# 如果文件夹名不太长，保留部分可读信息
	if len(edge_folder) <= 30:
		edge_identifier = edge_folder
	else:
		# 文件夹名太长，使用前15字符+hash
		edge_identifier = edge_folder[:15] + '_' + folder_hash
	
	cifname = f"{topo}_{node_short}_{edge_identifier}_{edge_count}.cif"
	
	# 确保文件名不超过200字符
	if len(cifname) > 200:
		# 如果还是太长，只使用topo+hash+count
		cifname = f"{topo}_{folder_hash}_{edge_count}.cif"
	
	return cifname

def run_template_with_assignment(edge_assignment):
	"""
	使用指定的edge assignment运行模板拼接
	"""
	start_time = time.time()
	TIMEOUT = 180  # 或使用 FOLDER_TIMEOUT
	
	# 新增：超时检查函数
	def check_timeout(location=""):
		elapsed = time.time() - start_time
		if elapsed > TIMEOUT:
			error_msg = f'Folder {folder_name} processing timed out after {elapsed:.1f}s at {location}'
			raise TimeoutError(error_msg)
	
	folder_name = edge_assignment['folder']
	check_timeout("start")
	print()
	print('=' * 100)
	print(f'PROCESSING FOLDER: {folder_name}')
	print('=' * 100)
	
	template_file = edge_assignment['template_file']
	if not template_file:
		print(f"⚠ SKIPPED: No template file found for topology {edge_assignment['topology']}")
		with open('skipped_folders.txt', 'a', encoding='utf-8') as f:
			f.write(f"Folder: {folder_name}\n")
			f.write(f"Reason: No template file found for topology {edge_assignment['topology']}\n")
			f.write('-' * 80 + '\n')
		return 0
	
	template_path = os.path.join('templates', template_file)
	if not os.path.exists(template_path):
		print(f"⚠ SKIPPED: Template file {template_path} not found")
		with open('skipped_folders.txt', 'a', encoding='utf-8') as f:
			f.write(f"Folder: {folder_name}\n")
			f.write(f"Reason: Template file {template_path} not found\n")
			f.write('-' * 80 + '\n')
		return 0
	
	print()
	print('Edge Assignment Folder:', edge_assignment['folder'])
	print('Template:', template_file)
	print('Topology:', edge_assignment['topology'])
	print('Metal Node List:', edge_assignment['metal_node_list'])
	print('Number of CIF files:', len(edge_assignment['cif_files']))
	print()
	
	metal_node_list = edge_assignment['metal_node_list']
	matching_nodes = find_matching_nodes(metal_node_list)

	if not matching_nodes:
		print(f"⚠ SKIPPED: No matching nodes found for {metal_node_list}")
		with open('skipped_folders.txt', 'a', encoding='utf-8') as f:
			f.write(f"Folder: {folder_name}\n")
			f.write(f"Reason: No matching nodes found for {metal_node_list}\n")
			f.write('-' * 80 + '\n')
		return 0

	print(f"✓ Using node file(s): {matching_nodes}")
	print()

	cat_count = 0
	for net in ct2g(template_file):
		check_timeout("net iteration")
		cat_count += 1
		TG, start, unit_cell, TVT, TET, TNAME, a, b, c, ang_alpha, ang_beta, ang_gamma, max_le, catenation = net

		TVT = sorted(TVT, key=lambda x:x[0], reverse=True)
		TET = sorted(TET, reverse=True)

		node_cns = [(cncalc(node, 'nodes'), node) for node in os.listdir('nodes')]

		print('Number of vertices = ', len(TG.nodes()))
		print('Number of edges = ', len(TG.edges()))
		print()

		edge_counts = dict((data['type'],0) for e0,e1,data in TG.edges(data=True))
		for e0,e1,data in TG.edges(data=True):
			edge_counts[data['type']] += 1
		
		if PRINT:
	
			print('There are', len(TG.nodes()), 'vertices in the voltage graph:')
			print()
			v = 0
	
			for node in TG.nodes():
				v += 1
				print(v,':',node)
				node_dict = TG.nodes[node]
				print('type : ', node_dict['type'])
				print('cartesian coords : ', node_dict['ccoords'])
				print('fractional coords : ', node_dict['fcoords'])
				print('degree : ', node_dict['cn'][0])
				print()
	
			print('There are', len(TG.edges()), 'edges in the voltage graph:')
			print()
	
			for edge in TG.edges(data=True,keys=True):
				edge_dict = edge[3]
				ind = edge[2]
				print(ind,':',edge[0],edge[1])
				print('length : ',edge_dict['length'])
				print('type : ',edge_dict['type'])
				print('label : ',edge_dict['label'])
				print('positive direction :',edge_dict['pd'])
				print('cartesian coords : ',edge_dict['ccoords'])
				print('fractional coords : ',edge_dict['fcoords'])
				print()
	
		vas, skip_reason = create_vertex_assignments(TG, TVT, matching_nodes, folder_name)

		if skip_reason:
			print(f"⚠ SKIPPED: {skip_reason}")
			with open('skipped_folders.txt', 'a', encoding='utf-8') as f:
				f.write(f"Folder: {folder_name}\n")
				f.write(f"Reason: {skip_reason}\n")
				f.write(f"Topology vertex types: {len(TVT)}\n")
				f.write(f"Provided nodes: {len(matching_nodes)} - {matching_nodes}\n")
				f.write('-' * 80 + '\n')
			return
		
		CB,CO = cycle_cocyle(TG)

		for va in vas:
			if len(va) == 0:
				print('At least one vertex does not have a building block with the correct number of connection sites.')
				print('Moving to the next template...')
				print()
				continue
	
		if len(CB) != (len(TG.edges()) - len(TG.nodes()) + 1):
			print('The cycle basis is incorrect.')
			print('The number of cycles in the cycle basis does not equal the rank of the cycle space.')
			print('Moving to the next tempate...')
			continue
		
		num_edges = len(TG.edges())
		Bstar, alpha = Bstar_alpha(CB,CO,TG,num_edges)

		if PRINT:
			print('B* (top) and alpha (bottom) for the barycentric embedding are:')
			print()
			for i in Bstar:
				print(i)
			print()
			for i in alpha:
				print(i)
			print()
	
		num_vertices = len(TG.nodes())
	
		# 使用指定的edge文件 - 构建相对路径
		ea_base = [os.path.join(edge_assignment['folder'], cif) for cif in edge_assignment['cif_files']]
		
		# 根据TET的长度和可用的edge文件数量生成组合
		num_edge_types = len(TET)
		num_available_cifs = len(ea_base)
		
		print(f"Topology has {num_edge_types} edge types, {num_available_cifs} CIF files provided")
		
		# 替换为：
		if num_edge_types > 5:  # 放宽限制到5
			print(f"⚠ SKIPPED: Topology has {num_edge_types} edge types (>5), skipping this folder")
			with open('skipped_folders.txt', 'a', encoding='utf-8') as f:
				f.write(f"Folder: {edge_assignment['folder']}\n")
				f.write(f"Reason: Topology has {num_edge_types} edge types (>5)\n")
				f.write('-' * 80 + '\n')
			return 0

		# 然后修改边的组合生成逻辑：
		if num_available_cifs == 1:
			# 1个边文件，用于所有边类型
			eas = [[ea_base[0]] * num_edge_types]
			print(f'1 edge file for {num_edge_types} edge types, reusing for all (1 combination)')
			
		elif num_available_cifs == 2:
			if num_edge_types == 1:
				eas = [[ea_base[0]]]
			elif num_edge_types == 2:
				eas = [ea_base]
			elif num_edge_types == 3:
				# [A, A, B] - 复用第一个
				eas = [[ea_base[0], ea_base[0], ea_base[1]]]
			elif num_edge_types == 4:
				# [A, A, B, B] - 均衡复用
				eas = [[ea_base[0], ea_base[0], ea_base[1], ea_base[1]]]
			print(f'2 edge files for {num_edge_types} edge types (reusing)')
			
		elif num_available_cifs == 3:
			if num_edge_types <= 3:
				eas = [[ea_base[i] for i in range(num_edge_types)]]
			elif num_edge_types == 4:
				# [A, B, C, A] - 复用第一个（通常最短）
				eas = [[ea_base[0], ea_base[1], ea_base[2], ea_base[0]]]
			elif num_edge_types == 5:
				# [A, B, C, A, B] - 复用前两个
				eas = [[ea_base[0], ea_base[1], ea_base[2], ea_base[0], ea_base[1]]]
			print(f'3 edge files for {num_edge_types} edge types (reusing)')
			
		else:
			# 3个以上的CIF文件
			if num_edge_types <= num_available_cifs:
				eas = [[ea_base[i] for i in range(num_edge_types)]]
			else:
				# 循环复用
				eas = [[ea_base[i % num_available_cifs] for i in range(num_edge_types)]]
			print(f'{num_available_cifs} edge files for {num_edge_types} edge types')
	
		g = 0

		skip_vertex = False

		# 遍历所有vertex assignments
		for va in vas:
			check_timeout("vertex assignment")
			node_elems = [bbelems(i[1], 'nodes') for i in va]
			metals = [[i for i in j if i in metal_elements] for j in node_elems]
			metals = list(set([i for j in metals for i in j]))
			
			v_set = [('v' + str(vname_dict[re.sub('[0-9]','',i[0])]), i[1]) for i in va]
			v_set = sorted(list(set(v_set)), key=lambda x: x[0])
			v_set = [v[0] + '-' + v[1] for v in v_set]
	
			print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
			print('vertex assignment : ',v_set)
			print('Meta.txt specified nodes:', matching_nodes)
			print('Assignment strategy:', 
				'Single node for all vertices' if len(matching_nodes) == 1 
				else f'{len(matching_nodes)} nodes distributed to {len(TVT)} vertex types')
			print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
			print()

			if SINGLE_METAL_MOFS_ONLY and len(metals) != 1:
				print(v_set, 'contains no metals or multiple metal elements, no cif will be written')
				print()
				with open('skipped_folders.txt', 'a', encoding='utf-8') as f:
					f.write(f"Folder: {edge_assignment['folder']}\n")
					f.write(f"Reason: SINGLE_METAL_MOFS_ONLY filter - contains {len(metals)} metal types\n")
					f.write(f"Metals found: {metals}\n")
					f.write(f"Vertex assignment: {v_set}\n")
					f.write('-' * 80 + '\n')
				continue

			if MOFS_ONLY and len(metals) < 1:
				print(v_set, 'contains no metals, no cif will be written')
				print()
				with open('skipped_folders.txt', 'a', encoding='utf-8') as f:
					f.write(f"Folder: {edge_assignment['folder']}\n")
					f.write(f"Reason: MOFS_ONLY filter - no metals found\n")
					f.write(f"Vertex assignment: {v_set}\n")
					f.write('-' * 80 + '\n')
				continue

			# 为每个顶点设置cifname（从va中获取）
			for v in va:
				for n in TG.nodes(data=True):
					if v[0] == n[0]:
						n[1]['cifname'] = v[1]  # 直接使用va中分配的node文件名		

			skip_vertex = False
			
			for ea in eas:
				check_timeout("edge assignment")
				g += 1
	
				print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
				print(f'Processing edge combination {g} for folder: {edge_assignment["folder"]}')
				print('edge assignment : ',ea)
				print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
				print()
				
				# 为每条边分配edge文件
				type_assign = dict((k,[]) for k in sorted(TET, reverse=True))
				for k,m in zip(TET,ea):
					type_assign[k] = m
		
				for e in TG.edges(data=True):
					ty = e[2]['type']
					for k in type_assign:
						if ty == k or (ty[1],ty[0]) == k:
							e[2]['cifname'] = type_assign[k]
				
				# 计算可能的X-X键数量
				num_possible_XX_bonds = 0
				for edge_type, cifname in zip(TET, ea):
					if 'ntn_edge' in cifname:
						factor = 1
					else:
						factor = 2
					edge_type_count = edge_counts[edge_type]
					num_possible_XX_bonds += factor * edge_type_count
	
				# 添加connection point mismatch错误处理
				check_timeout("before assign_node_vecs2edges")
				try:
					ea_dict, x_selection_info = assign_node_vecs2edges(TG, unit_cell, SYMMETRY_TOL, template_file)
				except ValueError as e:
					print('*******************************************')
					print('ERROR: Connection point mismatch')
					print('*******************************************')
					print(str(e))
					print()
					print('This error indicates that the node CIF file does not match the topology requirements.')
					print('Skipping this folder and moving to next assignment...')
					print()
					with open('skipped_folders.txt', 'a', encoding='utf-8') as f:
						f.write(f"Folder: {edge_assignment['folder']}\n")
						f.write(f"Reason: Connection point mismatch\n")
						f.write(f"Error: {str(e)}\n")
						f.write(f"Node files: {matching_nodes}\n")
						f.write(f"Template: {template_file}\n")
						f.write('-' * 80 + '\n')
					return  # 跳过整个文件夹，返回处理下一个
				check_timeout("before scaling")
				all_SBU_coords = SBU_coords(TG, ea_dict, CONNECTION_SITE_BOND_LENGTH)
				sc_a, sc_b, sc_c, sc_alpha, sc_beta, sc_gamma, sc_covar, Bstar_inv, max_length, callbackresults, ncra, ncca, scaling_data = scale(all_SBU_coords,a,b,c,ang_alpha,ang_beta,ang_gamma,max_le,num_vertices,Bstar,alpha,num_edges,FIX_UC,SCALING_ITERATIONS,PRE_SCALE,MIN_CELL_LENGTH,OPT_METHOD)
	
				print('*******************************************')
				print('The scaled unit cell parameters are : ')
				print('*******************************************')
				print('a    :', np.round(sc_a, 5))
				print('b    :', np.round(sc_b, 5))
				print('c    :', np.round(sc_c, 5))
				print('alpha:', np.round(sc_alpha, 5))
				print('beta :', np.round(sc_beta, 5))
				print('gamma:', np.round(sc_gamma, 5))
				print()
	
				for sc, name in zip((sc_a, sc_b, sc_c), ('a', 'b', 'c')):
					cflag = False
					if sc == MIN_CELL_LENGTH:
						print('unit cell parameter', name, 'may have collapsed during scaling!')
						print('try re-running with', name, 'fixed or a larger MIN_CELL_LENGTH')
						print('no cif will be written')
						cflag = True
	
				if cflag:
					continue
	
				scaled_params = [sc_a,sc_b,sc_c,sc_alpha,sc_beta,sc_gamma]
			
				sc_Alpha = np.r_[alpha[0:num_edges-num_vertices+1,:], sc_covar]
				sc_omega_plus = np.dot(Bstar_inv, sc_Alpha)
			
				ax = sc_a
				ay = 0.0
				az = 0.0
				bx = sc_b * np.cos(sc_gamma * pi/180.0)
				by = sc_b * np.sin(sc_gamma * pi/180.0)
				bz = 0.0
				cx = sc_c * np.cos(sc_beta * pi/180.0)
				cy = (sc_c * sc_b * np.cos(sc_alpha * pi/180.0) - bx * cx) / by
				cz = (sc_c ** 2.0 - cx ** 2.0 - cy ** 2.0) ** 0.5
				sc_unit_cell = np.asarray([[ax,ay,az],[bx,by,bz],[cx,cy,cz]]).T
				
				scaled_coords = omega2coords(start, TG, sc_omega_plus, (sc_a,sc_b,sc_c,sc_alpha,sc_beta,sc_gamma), num_vertices, template_file, g, WRITE_CHECK_FILES)
				nvecs,evecs = scaled_node_and_edge_vectors(scaled_coords, sc_omega_plus, sc_unit_cell, ea_dict)
				check_timeout("before place_nodes")
				try:
					placed_nodes, node_bonds = place_nodes(nvecs, CHARGES, ORIENTATION_DEPENDENT_NODES, x_selection_info)
				except (KeyError, IndexError, ValueError, AttributeError) as e:
					print('*******************************************')
					print('ERROR during node placement:', str(e))
					print('*******************************************')
					print('This error may be caused by:')
					print('  - Invalid atom labels in node CIF file')
					print('  - Atom indices exceeding dictionary range')
					print('  - Malformed or incompatible node structure')
					print()
					print('Problematic vertex assignment:', v_set)
					print('This vertex assignment has node placement issues.')
					print('Skipping ALL edge assignments for this vertex and moving to next vertex assignment...')
					print()
					with open('skipped_folders.txt', 'a', encoding='utf-8') as f:
						f.write(f"Folder: {edge_assignment['folder']}\n")
						f.write(f"Reason: Node placement error\n")
						f.write(f"Error: {str(e)}\n")
						f.write(f"Node files: {matching_nodes}\n")
						f.write(f"Vertex assignment: {v_set}\n")
						f.write('-' * 80 + '\n')
					skip_vertex = True
					break
				check_timeout("before place_edges")
				try:
					placed_edges, edge_bonds = place_edges(evecs, CHARGES, len(placed_nodes))
				except (KeyError, IndexError, ValueError, AttributeError) as e:
					print('*******************************************')
					print('ERROR during edge placement:', str(e))
					print('*******************************************')
					print('This error may be caused by:')
					print('  - Invalid atom labels in edge CIF file (e.g., C50, C100)')
					print('  - Atom indices exceeding dictionary range')
					print('  - Malformed or incompatible edge structure')
					print('  - Edge CIF file with unusual atom naming conventions')
					print()
					print('Problematic edge file(s):', ea)
					print('Skipping this edge assignment and moving to next configuration...')
					print()
					with open('skipped_folders.txt', 'a', encoding='utf-8') as f:
						f.write(f"Folder: {edge_assignment['folder']}\n")
						f.write(f"Reason: Edge placement error\n")
						f.write(f"Error: {str(e)}\n")
						f.write(f"Edge files: {ea}\n")
						f.write('-' * 80 + '\n')
					continue
	
				if RECORD_CALLBACK:
	
					vnames = '_'.join([v.split('.')[0] for v in v_set])
	
					if len(ea) <= 5:
						enames = '_'.join([e[0:-4] for e in ea])
					else:
						enames = str(len(ea)) + '_edges'
	
					prefix = template_file[0:-4] + '_' +  vnames + '_' + enames
	
					frames = scaling_callback_animation(callbackresults, alpha, Bstar_inv, ncra, ncca, num_vertices, num_edges, TG, template_file, g, False)
					write_scaling_callback_animation(frames, prefix)
					animate_objective_minimization(callbackresults, prefix)
	
				if PLACE_EDGES_BETWEEN_CONNECTION_POINTS:
					placed_edges = adjust_edges(placed_edges, placed_nodes, sc_unit_cell)
				
				placed_nodes = np.c_[placed_nodes, np.array(['node' for i in range(len(placed_nodes))])]
				placed_edges = np.c_[placed_edges, np.array(['edge' for i in range(len(placed_edges))])]

				placed_all = list(placed_nodes) + list(placed_edges)
				bonds_all = node_bonds + edge_bonds
		
				if WRITE_CHECK_FILES:
					write_check_cif(template_file, placed_nodes, placed_edges, g, scaled_params, sc_unit_cell)
		
				if REMOVE_DUMMY_ATOMS:
					placed_all, bonds_all, nconnections = remove_Fr(placed_all,bonds_all)
				else:
					nconnections = []
				check_timeout("before bond formation")
				print('computing X-X bonds...')
				print()
				print('*******************************************')
				print('Bond formation : ')
				print('*******************************************')
				
				try:
					fixed_bonds, nbcount, bond_check_passed = bond_connected_components(placed_all, bonds_all, sc_unit_cell, max_length, BOND_TOL, nconnections, num_possible_XX_bonds)
					print('there were ', nbcount, ' X-X bonds formed')
						
					if bond_check_passed:
						print('bond check passed')
						bond_check_code = ''
					else:
						print('bond check failed, attempting distance search bonding...')
						fixed_bonds, nbcount = distance_search_bond(placed_all, bonds_all, sc_unit_cell, 2.5)
						bond_check_code = '_BOND_CHECK_FAILED'
						print('there were', nbcount, 'X-X bonds formed')
					print()
				except ValueError as e:
					print('ERROR during bond formation:', str(e))
					print('Skipping this structure and moving to next configuration...')
					print()
					with open('skipped_folders.txt', 'a', encoding='utf-8') as f:
						f.write(f"Folder: {edge_assignment['folder']}\n")
						f.write(f"Reason: Bond formation error\n")
						f.write(f"Error: {str(e)}\n")
						f.write('-' * 80 + '\n')
					continue
		
				if CHARGES:
					fc_placed_all, netcharge, onetcharge, rcb = fix_charges(placed_all)
				else:
					fc_placed_all = placed_all
	
				fixed_bonds = fix_bond_sym(fixed_bonds, placed_all, sc_unit_cell)
	
				if CHARGES:
					print('*******************************************')
					print('Charge information :                       ')
					print('*******************************************')
					print('old net charge                  :', np.round(onetcharge, 5))
					print('rescaling magnitude             :', np.round(rcb, 5))

					remove_net = choice(range(len(fc_placed_all)))
					fc_placed_all[remove_net][4] -= np.round(netcharge, 4)

					print('new net charge (after rescaling):', np.sum([li[4] for li in fc_placed_all]))
					print()

				# 生成简短的文件名
				# 如果有多个node，使用第一个或组合名称
				if len(matching_nodes) == 1:
					node_name_for_cif = matching_nodes[0]
				else:
					# 多个node时，组合名称或使用第一个
					node_name_for_cif = '_'.join([n.replace('.cif', '') for n in matching_nodes])
					# 如果太长，只用第一个
					if len(node_name_for_cif) > 50:
						node_name_for_cif = matching_nodes[0]

				cifname = generate_short_cifname(
					template_file,
					node_name_for_cif,
					edge_assignment['folder'],
					g
				)
				
				if catenation:
					cifname = cifname.replace('.cif', f'_CAT{cat_count}.cif')
				check_timeout("before write_cif")
				if WRITE_CIF:
					print('writing cif...')
					print()
					if len(cifname) > 255:
						cifname = cifname[0:241]+'_truncated.cif'
					write_cif(fc_placed_all, fixed_bonds, scaled_params, sc_unit_cell, cifname, CHARGES)
					print(f"✓ Successfully generated: {cifname}")
					
					# 记录成功生成的文件
					with open('successful_outputs.txt', 'a', encoding='utf-8') as f:
						f.write(f"Folder: {edge_assignment['folder']}\n")
						f.write(f"Output: {cifname}\n")
						f.write(f"Combination: {g}\n")
						f.write('-' * 80 + '\n')

		if skip_vertex:
			continue

	if catenation and MERGE_CATENATED_NETS:
		
		print('merging catenated cifs...')
		cat_cifs = glob.glob('output_cifs/*_CAT*.cif')

		for comb in itertools.combinations(cat_cifs, cat_count):

			builds = [name[0:-9] for name in comb]

			print(set(builds))

			if len(set(builds)) == 1:
				pass
			else:
				continue

			merge_catenated_cifs(comb, CHARGES)

		for cif in cat_cifs:
			os.remove(cif)

def run_tobacco_with_edge_folders(CHARGES):
	"""
	主函数：扫描edges目录下的所有子文件夹并运行拼接
	"""
	# 初始化跟踪文件
	with open('skipped_folders.txt', 'w', encoding='utf-8') as f:
		f.write('Skipped Folders During Processing\n')
		f.write('=' * 80 + '\n\n')
	
	with open('successful_outputs.txt', 'w', encoding='utf-8') as f:
		f.write('Successfully Generated CIF Files\n')
		f.write('=' * 80 + '\n\n')
	
	edge_assignments, failed_assignments = get_edge_folder_assignments()
	
	# 写入失败的assignments到文件
	if failed_assignments:
		with open('failed_assignments.txt', 'w', encoding='utf-8') as f:
			f.write('Failed Edge Assignments\n')
			f.write('=' * 80 + '\n\n')
			for assignment in failed_assignments:
				f.write(f"Folder: {assignment['folder']}\n")
				f.write(f"Topology: {assignment['topology']}\n")
				f.write(f"Metal Node List: {assignment['metal_node_list']}\n")
				f.write(f"CIF Files: {len(assignment['cif_files']) if assignment['cif_files'] else 0}\n")
				f.write(f"Template File: {assignment['template_file']}\n")
				f.write(f"Issues:\n")
				for issue in assignment['issues']:
					f.write(f"  - {issue}\n")
				f.write('\n' + '-' * 80 + '\n\n')
		print(f"\n{len(failed_assignments)} failed assignments written to failed_assignments.txt\n")
	
	if not edge_assignments:
		print("No valid edge assignments found in edges directory!")
		return 0
	
	print(f"\nProcessing {len(edge_assignments)} valid edge assignments\n")
	
	processing_failures = []
	
	if IGNORE_ALL_ERRORS:
		for assignment in edge_assignments:
			try:
				run_template_with_assignment(assignment)
			
			# 新增：捕获超时异常
			except TimeoutError as e:
				print()
				print('*' * 80)
				print(f'⏱ TIMEOUT: {str(e)}')
				print('*' * 80)
				print()
				
				processing_failures.append({
					'folder': assignment['folder'],
					'topology': assignment['topology'],
					'metal_node': assignment['metal_node'],
					'error': f'Timeout: {str(e)}'
				})
				
				with open('skipped_folders.txt', 'a', encoding='utf-8') as f:
					f.write(f"Folder: {assignment['folder']}\n")
					f.write(f"Reason: {str(e)}\n")
					f.write('-' * 80 + '\n')
			
			except Exception as e:
				print()
				print('*****************************************************************')
				print('ERROR for edge assignment:', assignment['folder'])      
				print('error message:', e)
				print('continuing to next assignment...')                          
				print('*****************************************************************')
				print()
				processing_failures.append({
					'folder': assignment['folder'],
					'topology': assignment['topology'],
					'metal_node': assignment['metal_node'],
					'error': str(e)
				})
	else:
		for assignment in edge_assignments:
			try:
				run_template_with_assignment(assignment)
			
			# 新增：捕获超时异常（即使IGNORE_ALL_ERRORS=False也要记录）
			except TimeoutError as e:
				print()
				print('*' * 80)
				print(f'⏱ TIMEOUT: {str(e)}')
				print('*' * 80)
				print()
				
				processing_failures.append({
					'folder': assignment['folder'],
					'topology': assignment['topology'],
					'metal_node': assignment['metal_node'],
					'error': f'Timeout: {str(e)}'
				})
				
				with open('skipped_folders.txt', 'a', encoding='utf-8') as f:
					f.write(f"Folder: {assignment['folder']}\n")
					f.write(f"Reason: {str(e)}\n")
					f.write('-' * 80 + '\n')
				
				# 根据配置决定是否继续
				if not IGNORE_ALL_ERRORS:
					raise
			
			except Exception as e:
				processing_failures.append({
					'folder': assignment['folder'],
					'topology': assignment['topology'],
					'metal_node': assignment['metal_node'],
					'error': str(e)
				})
				raise
	
	# 写入处理失败的assignments
	if processing_failures:
		with open('processing_failures.txt', 'w', encoding='utf-8') as f:
			f.write('Processing Failures\n')
			f.write('=' * 80 + '\n\n')
			for failure in processing_failures:
				f.write(f"Folder: {failure['folder']}\n")
				f.write(f"Topology: {failure['topology']}\n")
				f.write(f"Metal Node List: {failure['metal_node_list']}\n")
				f.write(f"Error: {failure['error']}\n")
				f.write('\n' + '-' * 80 + '\n\n')
		print(f"\n{len(processing_failures)} processing failures written to processing_failures.txt\n")
	
	# 生成完整的统计报告
	print('\n' + '=' * 100)
	print('FINAL STATISTICS')
	print('=' * 100)
	
	total_folders = len(edge_assignments) + len(failed_assignments)
	print(f"Total input folders scanned: {total_folders}")
	print(f"  - Failed to parse (in failed_assignments.txt): {len(failed_assignments)}")
	print(f"  - Valid for processing: {len(edge_assignments)}")
	print(f"  - Processing failures (in processing_failures.txt): {len(processing_failures)}")
	print(f"  - Successfully processed: {len(edge_assignments) - len(processing_failures)}")
	
	# 统计成功输出的CIF文件数量
	try:
		with open('successful_outputs.txt', 'r', encoding='utf-8') as f:
			content = f.read()
			num_outputs = content.count('Output:')
		print(f"\nTotal CIF files generated: {num_outputs}")
	except:
		print(f"\nTotal CIF files generated: 0")
	
	# 统计跳过的文件夹
	try:
		with open('skipped_folders.txt', 'r', encoding='utf-8') as f:
			content = f.read()
			num_skipped = content.count('Folder:')
		print(f"Total folders skipped during processing: {num_skipped}")
	except:
		num_skipped = 0
		print(f"Total folders skipped during processing: 0")
	
	print('\n' + '=' * 100)
	print('OUTPUT FILES:')
	print('  - successful_outputs.txt : List of all generated CIF files')
	print('  - failed_assignments.txt : Folders that failed meta.txt parsing')
	print('  - skipped_folders.txt : Folders skipped during processing')
	print('  - processing_failures.txt : Folders with processing errors')
	print('=' * 100)
	
	# 如果有缺失，给出详细建议
	missing_count = total_folders - num_outputs - len(failed_assignments) - num_skipped - len(processing_failures)
	if missing_count > 0:
		print('\n⚠ WARNING: Some folders may not have generated outputs.')
		print('Please check the following files for details:')
		print('  1. failed_assignments.txt - for parsing issues')
		print('  2. skipped_folders.txt - for missing templates/nodes')
		print('  3. processing_failures.txt - for processing errors')
		print('  4. Check if SINGLE_METAL_MOFS_ONLY or MOFS_ONLY filters excluded some structures')

	
	# 统计输出结果
	output_cifs = glob.glob('output_cifs/*.cif')
	num_outputs = len(output_cifs)
	
	print()
	print('=' * 100)
	print('SUMMARY')
	print('=' * 100)
	print(f"Failed parsing (see failed_assignments.txt): {len(failed_assignments)}")
	print(f"Valid for processing: {len(edge_assignments)}")
	print(f"Processing failures (see processing_failures.txt): {len(processing_failures)}")
	print(f"Successfully processed: {len(edge_assignments) - len(processing_failures)}")
	print(f"Output CIF files generated: {num_outputs}")
	
	discrepancy = len(edge_assignments) - len(processing_failures) - num_outputs
	if discrepancy != 0:
		print(f"⚠ WARNING: Processed {len(edge_assignments) - len(processing_failures)} folders but only generated {num_outputs} outputs!")
		print(f"  Possible reasons: skipped due to missing node/template, or multiple combinations per folder")
	
	print('=' * 100)
	if len(failed_assignments) == 0:
		return 1
	else:
		return 0


if __name__ == '__main__':

	start_time = time.time()
	
	# 创建输出目录
	if not os.path.exists('output_cifs'):
		os.makedirs('output_cifs')
	
	if WRITE_CHECK_FILES and not os.path.exists('check_cifs'):
		os.makedirs('check_cifs')
	
	for d in ['templates', 'nodes', 'edges']:
		try:
			os.remove(os.path.join(d,'.DS_Store'))
		except:
			pass
	
	# 使用新的edge folder扫描方式
	run_tobacco_with_edge_folders(CHARGES)
	
	print('Normal termination of Tobacco_4.0 after')
	print('--- %s seconds ---' % (time.time() - start_time))
	print()
