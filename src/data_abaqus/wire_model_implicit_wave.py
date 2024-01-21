"""
Script for creating and analyzing a 3D beam model with multiple material sections using Abaqus.

This script defines functions to generate a wire, partition it into sections, assign materials,
create a rectangular profile, and generate a 3D beam model with specified materials for each section.
It also sets up boundary conditions, loads, and meshes the model before creating and submitting an Abaqus job.

Functions:
- create_wire(model, part, length)
- create_datum_plane_by_principal(model, part, type_plane, offset_plane)
- create_partition_by_plane(model, part, id_plane)
- create_material_elastic(model, material, E, nu)
- create_rectangular_profile(model, profile, a, b)
- create_beam_section(model, section, profile, material)
- create_set_coordinates(model, part, set_name, x, y, z)
- create_set_vertex(model, instance, x, y, z, set_name="load_set")
- assign_section(model, part, set_name, section)
- assign_beam_orientation(model, part, set_name)
- generate_constrained_vector(length, max_value, max_unique_numbers=3)
- mapping(number)
- create_assembly(model, part, instance)
- create_step(model, step)
- assign_y_load(model, instance, step, load, set)
- assign_encastre(model, instance, step, bc, set)
- mesh_instance(model, instance, mesh_size)
- create_mesh(model, instance)
- create_job(job_name, model)
- SubmitJob(job_name)

Usage:
- Update the script with the desired parameters (e.g., material properties, dimensions).
- Run the script in an Abaqus Python environment.

Author: Lorenzo Miele
Date: 12/23
"""

from abaqus import *
from abaqusConstants import *
import regionToolset
import __main__
import section
import regionToolset
import part
import material
import assembly
import step
import interaction
import load
import mesh
import job
import sketch
import visualization
import xyPlot
import connectorBehavior
import odbAccess
from odbAccess import openOdb
from operator import add
import time
import numpy as np
import os
from sys import exit


MATERIALS_NUMBER = 3

############## FUNCTIONS ##############
def create_model(MyModel):
    mdb.Model(name = MyModel)

def create_wire(model, part, length):
	s = mdb.models[model].ConstrainedSketch(name='__profile__', 
		sheetSize=200.0)
	g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
	s.setPrimaryObject(option=STANDALONE)
	s.Line(point1=(0.0, 0.0), point2=(length, 0.0))
	p = mdb.models[model].Part(name=part, dimensionality=THREE_D, 
		type=DEFORMABLE_BODY)
	p = mdb.models[model].parts[part]
	p.BaseWire(sketch=s)
	s.unsetPrimaryObject()
	p = mdb.models[model].parts[part]
	del mdb.models[model].sketches['__profile__']

def create_datum_plane_by_principal(model, part, type_plane, offset_plane):
    p = mdb.models[model].parts[part]
    myPlane = p.DatumPlaneByPrincipalPlane(principalPlane=type_plane, offset=offset_plane)
    myID = myPlane.id
    return myID

def create_partition_by_plane(model,part,id_plane):
    p = mdb.models[model].parts[part]
    e = p.edges[:]
    d = p.datums
    p.PartitionEdgeByDatumPlane(datumPlane=d[id_plane], edges = e)

def create_material_elastic(model, material, E, nu, rho):
	mdb.models[model].Material(name=material)
	mdb.models[model].materials[material].Elastic(table=((E, nu), ))
	mdb.models[model].materials[material].Density(table=((rho, ), ))

def create_rectangular_profile(model, profile, a, b):
	mdb.models[model].RectangularProfile(name=profile, a=a, 
		b=b)

def create_beam_section(model, section, profile, material):
	mdb.models[model].BeamSection(name=section, 
		integration=DURING_ANALYSIS, poissonRatio=0.0, profile=profile, 
		material=material, temperatureVar=LINEAR, consistentMassMatrix=False)

def create_set_coordinates(model, part, set_name, x ,y ,z):
	p = mdb.models[model].parts[part]
	e = p.edges
	edges = e.findAt(((x, y, z),))
	p.Set(edges=edges, name=set_name)

def create_set_vertex(model, instance, x, y, z, set_name = "load_set"):
	a = mdb.models[model].rootAssembly
	v1 = a.instances[instance].vertices
	verts1 = v1.findAt(((x, y, z),))
	a.Set(vertices=verts1, name=set_name)
	return set_name

def assign_section(model, part, set_name, section):
	p = mdb.models[model].parts[part]
	region = p.sets[set_name]
	p.SectionAssignment(region=region, sectionName=section, offset=0.0, 
		offsetType=MIDDLE_SURFACE, offsetField='', 
		thicknessAssignment=FROM_SECTION)

def assign_beam_orientation(model, part, set_name):
	p = mdb.models[model].parts[part]
	region=p.sets[set_name]
	p.assignBeamSectionOrientation(region=region, method=N1_COSINES, n1=(0.0, 0.0, 
		-1.0))

def read_array_from_txt(file_path):
    print("Full path to array file:", file_path)
    return np.loadtxt(file_path)

def mapping(number):
    return mydict[number]

def get_part_weight(model, part):
	weight = mdb.models[model].parts[part].getMassProperties()['mass']
	return weight

def create_assembly(model, part, instance):
	a1 = mdb.models[model].rootAssembly
	p = mdb.models[model].parts[part]
	a1.Instance(name=instance, part=p, dependent=OFF)

def create_step(model, step):
	mdb.models[model].ImplicitDynamicsStep(name=step, previous='Initial', 
    timePeriod=0.01, initialInc=1e-06, minInc=1e-07, maxNumInc=1000000)

def create_field_output (model):
    # Creating this function to reduce the analysis time
    mdb.models[model].fieldOutputRequests['F-Output-1'].setValues(variables=(
    'UT', ))

def create_tabular_wave (model, table_name, data_wave):
    mdb.models[model].TabularAmplitude(name=table_name, timeSpan=STEP, 
    smooth=SOLVER_DEFAULT, data=(data_wave))

def get_data_from_wave_txt(file_path):
    tuple_list = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(', ')
            tuple_list.append((float(parts[0]), float(parts[1])))
    # remove the last 2 input because it is not monotonically increasing
    tuple_list = tuple_list[:-2]
    return tuple_list

def assign_y_wave_load(model, instance, step, load, set, table_name, input_path):
	data = get_data_from_wave_txt(input_path)
	print(data)
	create_tabular_wave(model, table_name, data)
	a = mdb.models[model].rootAssembly
	region = a.sets[set]
	mdb.models[model].ConcentratedForce(name=load,
		createStepName=step, region=region, cf2=1.0, distributionType=UNIFORM,
		field='', localCsys=None)
	mdb.models[model].loads[load].setValues(amplitude=table_name, 
    distributionType=UNIFORM, field='')

def assign_encastre(model, instance, step, bc, set):
	a = mdb.models[model].rootAssembly
	region = a.instances[instance].sets[set]
	mdb.models[model].EncastreBC(name=bc, createStepName=step,
		region=region, localCsys=None)

def mesh_instance(model, instance, mesh_size):
	a = mdb.models[model].rootAssembly
	partInstances =(a.instances[instance], )
	a.seedPartInstance(regions=partInstances, size=mesh_size, deviationFactor=0.1, 
		minSizeFactor=0.1)

def create_mesh(model, instance):
	a = mdb.models[model].rootAssembly
	partInstances =(a.instances[instance], )
	a.generateMesh(regions=partInstances)

def pick_node_from_mesh(model, instance, x, node_name):
    a = mdb.models[model].rootAssembly
    n1 = a.instances[instance].nodes
    a.Set(name=node_name, nodes=n1.sequenceFromLabels((n1.getClosest((x, 0, 0 )).label,)))

def create_field_output_from_node(model, step, node_name, field_name):
    regionDef=mdb.models[model].rootAssembly.sets[node_name]
    mdb.models[model].FieldOutputRequest(name=field_name, 
        createStepName=step, variables=('UT', ), region=regionDef, 
        sectionPoints=DEFAULT, rebar=EXCLUDE)

def create_job(job_name, model):
	mdb.Job(name=job_name, model=model, description='', type=ANALYSIS, 
		atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
		memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
		explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
		modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
		scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1, 
		numGPUs=0)

def SubmitJob(job_name):
    mdb.jobs[job_name].submit(consistencyChecking=OFF)

def Open_ODB_and_save(job_name, step,  node_name, treshold):
    f = open_odb_when_ready(job_name)
    print(f)
    if f is not None:
        f
        session.viewports['Viewport: 1'].setValues(displayedObject=f)
        xy_data_list = session.xyDataListFromField(odb=f, outputPosition=NODAL, variable=(('U', 
            NODAL, ((INVARIANT, 'Magnitude'), )), ), nodeSets=(node_name.upper(), ))
        # Time spent to find out abaqus use only upper string: 2.15h
        curve_data = xy_data_list[0].data
        # point[0] is the time, point[1] is the U2 magnitude
        max_y_value = max(point[1] for point in curve_data)
        for point in curve_data:
            x_value, y_value = point[0], point[1]
            if y_value > treshold * max_y_value:
                time_reached = x_value
                break
    else:
        print("Error opening the ODB file.")
        return None
    return time_reached

def open_odb_when_ready(job_name):
    odb_file = job_name + ".odb"
    max_wait_time = 300  # max waiting time
    wait_interval = 10  # interval
    elapsed_time = 0
    while os.path.exists(job_name + ".lck"):
        time.sleep(wait_interval)
        elapsed_time += wait_interval
        print("Elapsed time is:"+ str(elapsed_time))
        if elapsed_time >= max_wait_time:
            print('Timeout: The file odb was not generated in' + str(max_wait_time) + 'seconds')
            return None
    return session.openOdb(odb_file)

def write_output_to_txt(time_output, output_file_path):
    if os.path.exists(output_file_path):
        print("File Modified:" + (output_file_path))
        with open(output_file_path, 'a') as file:
            file.write("\nWave reach time: " + str(time_output))
    else:
        print("Error: File not Found")
        return None

############## MAIN ##############

# Set the current working directory to the script's folder
script_folder = os.getcwd()
print(script_folder)

# Define simulation parameters
MyModel = "Model-3"
MyPart = 'Beam'
MyInstance = 'Beam'
MyStep = 'MyWaveStep'
MyLoad = 'UpForce'
MyBC = 'Encastre'
MyMeshSize = 2.0
MyJob = "Beam_wave"
array_file_name = "array.txt"
output_file_name = "output_results.txt"
input_file_name = "Abaqus_input.txt"
Table = "Amplitude"

# Define paths for array and output files
folder_temporary_relative = os.path.join("data", "temporary")
array_file_path = os.path.join(script_folder, folder_temporary_relative, array_file_name)
input_wave_file_path = os.path.join(script_folder, folder_temporary_relative, input_file_name)
output_file_path = os.path.join(script_folder, folder_temporary_relative, output_file_name)

# Define geometric parameters
length = 1000
dimension = 100
dx = length/dimension

# Define material properties. Adding materials is possible
pet = "PET"
E_pet = 3500
nu_pet = 0.35
rho_pet = 0.0000000014
abs = "ABS"
E_abs = 2300
nu_abs = 0.4
rho_abs = 0.0000000011
pla = "PLA"
E_pla = 3200
nu_pla = 0.33
rho_pla = 0.0000000013

# Define beam profile parameters
profile = "square"
a = 10
b = 10

# Define section names
section_pet = "PET_section"
section_abs = "ABS_section"
section_pla = "PLA_section"

# Node for the feature extraction
node_name = 'pick_node'
field_name = 'Node_field'
node_x_coord = 300 # approximately to the 30% of the beam from the encastre

# Create the Model
create_model(MyModel)

# Create wire geometry
create_wire(MyModel, MyPart, length)

# Create datum planes for partitioning
for i in range(1, 100):
    ID = create_datum_plane_by_principal(MyModel, MyPart, YZPLANE, i*dx)
    create_partition_by_plane(MyModel,MyPart,ID)

# Create sets with coordinates for each partition
for i in range(100):
    create_set_coordinates(MyModel, MyPart, "set-"+ str(i), dx*(i-1/2) ,0 ,0)

# Create rectangular profile
create_rectangular_profile(MyModel, profile, a, b)

# Create elastic materials
create_material_elastic(MyModel, pet, E_pet, nu_pet, rho_pet)
create_material_elastic(MyModel, abs, E_abs, nu_abs, rho_abs)
create_material_elastic(MyModel, pla, E_pla, nu_pla, rho_pla)

# Create beam sections
create_beam_section(MyModel, section_pet, profile, pet)
create_beam_section(MyModel, section_abs, profile, abs)
create_beam_section(MyModel, section_pla, profile, pla)

# Read section vector from file
section_vector = read_array_from_txt(array_file_path)

# Map section vector to section names
mydict = {1: "PET_section", 2: "ABS_section", 3: "PLA_section"}
section_string_vector = [mydict[number] for number in section_vector]
print(max(section_string_vector))

# Assign sections to each set and define beam orientation
for i, section in enumerate(section_string_vector):
    set_name = "set-" + str(i)
    assign_section(MyModel, MyPart, set_name, section)
    assign_beam_orientation(MyModel, MyPart, set_name)

# Create assembly, step, and assign boundary conditions and loads
create_assembly(MyModel, MyPart, MyInstance)
# create_field_output (MyModel)
create_step(MyModel, MyStep)
load_set = create_set_vertex(MyModel, MyInstance, 1000, 0, 0)
data_wave = get_data_from_wave_txt(input_wave_file_path)
assign_y_wave_load(MyModel, MyInstance, MyStep, MyLoad, load_set, Table, input_wave_file_path)
assign_encastre(MyModel, MyInstance, MyStep, MyBC, "set-0")

# Mesh the instance and create the mesh
mesh_instance(MyModel, MyInstance, MyMeshSize)
create_mesh(MyModel, MyInstance)

# Create an output on a mesh node
pick_node_from_mesh(MyModel, MyInstance, node_x_coord, node_name)
create_field_output_from_node(MyModel, MyStep, node_name, field_name)

# Create and submit the job
create_job(MyJob, MyModel)
SubmitJob(MyJob)

# Extract weight and maximum stress from the simulation
time_output = Open_ODB_and_save(MyJob, MyStep, node_name, treshold = 0.2)

# Write output to the specified text file
write_output_to_txt(time_output, output_file_path)

# Exit the script
exit()