# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:34:53 2020

import KamodofyPlasmaPy as KPP

#to include the (almost) entire PlasmaPy function library:
kamodo = KPP.AllPlasmaPy()   # into a new kamodo object
kamodo = KPP.AllPlasmaPy(kamodo=kamodo)  # into an existing kamodo object
kamodo = KPP.AllPlasmaPy(verbose=1)   #to see more output

#to include a portion of the PlasmaPy function library:
kamodo = KPP.ListPlasmaPy([list of function names])   # into a new kamodo object
kamodo = KPP.ListPlasmaPy([list of function names], kamodo=kamodo)  # into an existing kamodo object
kamodo = KPP.ListPlasmaPy([list of function names], verbose=1)   #to see more output

@author: rebec
"""

#PlasmaPy wrapper code, written assuming annotations are included for each function

import itertools, importlib
from functools import partial, update_wrapper
from inspect import signature
from astropy import units as u
from KamodoMaster.kamodo.kamodo import kamodofy, Kamodo
from sympy.abc import _clash


#citation for plasmapy
PP_citation = 'PlasmaPy Community et al. (2021). PlasmaPy, version 0.5.0, Zenodo, http://doi.org/10.5281/zenodo.4602818'

#'plasmapy.formulary.magnetostatics' only returns class objects, excluding
module_list = ['plasmapy.diagnostics.langmuir', 'plasmapy.diagnostics.thomson',
               'plasmapy.formulary.braginskii', 'plasmapy.formulary.collisions', 
               'plasmapy.formulary.dielectric', 'plasmapy.formulary.dimensionless',  
               'plasmapy.formulary.drifts', 'plasmapy.formulary.ionization', 
               'plasmapy.formulary.parameters', 'plasmapy.formulary.quantum',
               'plasmapy.formulary.radiation', 'plasmapy.formulary.relativity', 
               'plasmapy.formulary.distribution', 'plasmapy.formulary.mathematics']
bad_function_list = ['swept_probe_analysis','Characteristic',
                     'get_plasma_potential','get_floating_potential',
                     'get_electron_saturation_current','get_ion_saturation_current',
                     'extract_exponential_section','extract_ion_section',
                     'get_electron_temperature','extrapolate_electron_current',
                     'get_ion_density_OML','extrapolate_ion_current_OML',
                     'get_EEDF','spectral_density','ClassicalTransport',
                     'impact_parameter','cold_plasma_permittivity_SDP',
                     'cold_plasma_permittivity_LRP']

def _Astropy(func):
    '''Retrieve value from AstroPy quantity in the function output'''
    
    #handle conversion to SI units per variable
    def astropy_wrapper(*args, **kwargs):
        try: 
            return func(*args, **kwargs).value
        except: 
            return func(*args, **kwargs)  #not all returns types are astropy quantities
    
    return update_wrapper(astropy_wrapper, func)

def _ToHz(to_hz, func):
    '''Apply to_hz value to function given, correcting the function signature.'''
    
    new_func = update_wrapper(partial(func, to_hz=to_hz), func)
    old_sig, par_list = signature(new_func), []
    for key in old_sig.parameters:  #retrieve list of parameters, excluding to_hz
        if key != 'to_hz': par_list.append(old_sig.parameters[key])
    if to_hz:  #also update output units to Hz
        new_func.__signature__ = old_sig.replace(return_annotation=u.Hz, parameters=par_list)
    else:  #keep rad/s output units
        new_func.__signature__ = old_sig.replace(parameters=par_list)
    return new_func


#Need to know what default value of to_hz is in function call, and correct output units if False
def _gettohz(func):
    '''Get default value for to_hz if in function call'''
    
    if 'to_hz' in signature(func).parameters:
        return signature(func).parameters['to_hz'].default
    else: #to_hz keyword not present in function call.
        return None
        
def _KamodoUnit(old_unit, func_name=''):
    '''Correct string output from astropy for Kamodo'''

    if not isinstance(old_unit, str): #convert to string, preferably using astropy
        try: 
            old_unit = old_unit.to_string('ogip')
        except:
            old_unit = str(old_unit)
    if (old_unit=='[Unit("1 / m3"), Unit("kg / m3")]'): 
        if func_name=='mass_density': return '1/m**3'  #allow n_density here for conversion
        else: return 'kg/m**3'  #enforce mass_density input only
    for bad in ['class', 'typing', 'Union']: 
        if bad in old_unit: return 'm/m'   #replace bad unit descriptors with dimensionless
    for item in ['S','N','Ohm']:  #force sympy to accept by decomposing the unit
        if item in old_unit: old_unit = u.Unit(old_unit).decompose().to_string('ogip')
    unit_list = [(' / ','/'),(' * ','*'),(' ','*')]  #cannot accept ^ as an operator
    for old, new in unit_list: old_unit = old_unit.replace(old,new)
    if 'radian' not in old_unit: old_unit = old_unit.replace('rad','radian')
    if old_unit=='': old_unit='m/m'  #kamodo represents 'm/m' better than ''
    
    return old_unit

def _PPunits(func, to_hz=None):  
    '''Automatically determine output units of function
    
    For wrapped partial functions generated by ToHz, 
        to_hz must be specified because it is no longer in the function signature.
    '''
    
    if to_hz is None: 
        value = _gettohz(func)
    else:
        value = to_hz
    if value==True: return 'Hz'
    if value==False: return 'radian/s'
    if value is None: 
        try:  #most should have return annotations
            unit_out = _KamodoUnit(func.__annotations__['return'].to_string('ogip')) #ogip
        except: #some returns are unitless or lack a return annotation
            unit_out = 'm/m'
    #print(f'return_unit for {func.__name__} is: ', unit_out)
    return unit_out

def _ArgDict(func, inf_gen):
    '''return dictionary of strings representing arg names and units'''
    
    #kamodo gets arg_names from func, not arg_units, so don't bother with _clash
    old_sig, arg_dict = signature(func), {}
    for key in old_sig.parameters.keys():
        arg_dict[key] = _KamodoUnit(old_sig.parameters[key].annotation,
                                    func.__name__)  #include func_name to check 
    if arg_dict=={}:
        print('No units found for any variables for', func)

    return arg_dict

def _KamodofyPlasmaPy(func, inf_gen, to_hz=False):
    '''Simplify kamodofication process of a PlasmaPy function.'''
    
    #some functions are being kamodofied more than once here. Fixing.
    K_test = False  #initialize kamodofication status variable
     
    #check for to_hz in function signature, return properly kamodofied function
    if ('to_hz' not in signature(func).parameters) and (not K_test):
        K_test = True
        return kamodofy(_Astropy(func), units=_PPunits(func), 
                        arg_units=_ArgDict(func, inf_gen), citation=PP_citation)    
    elif not K_test: 
        if not to_hz:
            K_test = True
            return kamodofy(_Astropy(_ToHz(False, func)), 
                                   units=_PPunits(func,to_hz=False), 
                                   arg_units=_ArgDict(func, inf_gen),
                                   hidden_args=['to_hz'], citation=PP_citation)
        else: 
            K_test = True
            return kamodofy(_Astropy(_ToHz(True, func)), 
                                   units=_PPunits(func,to_hz=True), 
                                   arg_units=_ArgDict(func, inf_gen),
                                   hidden_args=['to_hz'], citation=PP_citation)
    if not K_test: 
        print(func, "not kamodofied. Check logic of module.")
        return K_test

def _KamodoFuncName(func, inf_gen, kamodo):
    '''Return best option for function name in kamodo object'''
    
    #try pulling return name from documentation
    return_name = ''  #default value
    if len(func.__doc__.split('Returns'))>1:
        return_name = func.__doc__.split('Returns')[1].split('\n')[2].split(':'
                                        )[0].replace(' ','')
        if len(return_name.split(','))>1:  #var1,var2
            print('Error: The',func.__name__, 'function returns more than one value! Skipping function.')
            return None
        if 'astropy' in return_name: return_name = '' #avoid names with variable type name

    #for functions without a return name, try making one from the alias name
    if(return_name=='') and (len(func.__doc__.split('Aliases'))>1):
        alias = func.__doc__.split('Aliases')[1].split('`')[1][:-1]  #ignore trailing _
        if alias[-2].islower() and alias[-1].isupper():  #e.g. betaH -> beta_H
            return_name = alias[:-1]+'_'+alias[-1]
        else:
            return_name = alias
            
    #avoid empty or long names and overwriting existing kamodo functions or sympy variables
    if (return_name=='') or (len(return_name)>len(func.__name__)): return_name = func.__name__
    if (sum([str(k)==return_name for k, v in kamodo.items()])>0) or (return_name in _clash): 
        return_name+=str(next(inf_gen))
        
    return return_name, inf_gen

def _SinglePlasmaPy(func, inf_gen, kamodo, verbose=0):
    '''Kamodofy given PlasmaPy function'''
    
    return_name, inf_gen = _KamodoFuncName(func, inf_gen, kamodo)
    if return_name is None: 
        return kamodo
    
    #pull into kamodo object, separately for functions with to_hz keyword
    if verbose > 0: print('Importing', func.__name__, 'into kamodo object as', return_name, '.')
    if 'to_hz' not in signature(func).parameters:
        try:
            kamodo[return_name] = _KamodofyPlasmaPy(func, inf_gen) #tries again if error
        except:
            print('Return unit: ', _PPunits(func), 'Arg_Dict: ', _ArgDict(func, inf_gen))
            print('Error: Kamodofication of', func.__name__, 'as', return_name, 
                  'failed. Skipping function.\n')
            return kamodo
    else:
        if verbose > 0: print('-Two versions of',return_name,'will be generated: one in radian/s and one in Hz.')
        kamodo[return_name+'1'] = _KamodofyPlasmaPy(func, inf_gen, to_hz=False)
        kamodo[return_name+'2'] = _KamodofyPlasmaPy(func, inf_gen, to_hz=True)
    
    return kamodo

def AllPlasmaPy(kamodo=None, verbose=0):
    '''Pull all compatible plasmapy functions into a single kamodo object'''
    
    if kamodo is None: 
        kamodo = Kamodo()
        print("Adding all compatible PlasmaPy functions into a new Kamodo object.")
    else:
        print("Adding all compatible PlasmaPy functions into the Kamodo object given.")

    inf_gen = itertools.count(start=1, step=1)  #initialize number generator
    for module_string in module_list:
        if verbose > 1: print(f'\nModule: {module_string}\n'+50*'-')
        if verbose == 1: print('\nAdding from module', module_string)
        module = importlib.import_module(module_string)
        for item in module.__all__:
            if item[-1]!='_':  #ignore function copies
                if item in bad_function_list:
                    print(f"{module_string}.{item} is known to return incompatible data. Skipping.") 
                    continue
      
                #kamodofy each function
                func = getattr(module, item)
                kamodo = _SinglePlasmaPy(func, inf_gen, kamodo, verbose=verbose)
        
    print("Done.")
    return kamodo

def ListPlasmaPy(func_list, kamodo=None, verbose=0):
    '''given list of desired functions (strings) in plasmapy, kamodofy each and return'''

    #initialize needed parameters
    if kamodo is None: #initialize a kamodo object if not given
        kamodo = Kamodo()
        print("Adding given PlasmaPy functions into a new Kamodo object.")
    else:
        print("Adding given PlasmaPy functions into the Kamodo object given.")
    if isinstance(func_list, str): func_list = [func_list]  #convert a single string to a list
    inf_gen = itertools.count(start=1, step=1)  #initialize number generator for var naming   
    
    #add each function
    for item in func_list:
        #separate function name from module name
        item_arr = item.split('.')
        func_string = item_arr[-1]
        module_string = item_arr[0]
        for piece in item_arr[1:-1]: module_string+='.'+piece
        
        #ignore bad functions or modules
        if (module_string not in module_list) or (func_string in bad_function_list):
            print(item,
                'is known to return incompatible data, or is typed incorrectly. Skipping.')
            continue   #skip to next function
        else: 
            if verbose > 0: print(f"Adding {module_string}.{func_string} to kamodo object.")
        
        #kamodofy each function
        module = importlib.import_module(module_string)
        func = getattr(module, func_string)
        kamodo = _SinglePlasmaPy(func, inf_gen, kamodo, verbose=verbose)       

    print("Done.")
    return kamodo