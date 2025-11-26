"""
Chemistry Lab Benchmark - Tools and Validators

This module provides chemistry calculation tools and validators for testing
CaveAgent's ability to perform chemical calculations and logical reasoning.
"""

from typing import Dict, List, Any, Tuple
from cave_agent.python_runtime import Variable, PythonRuntime
from core.validation import ValidatorResult
from core.types import BenchmarkTurn, ToolCall
import math


# ============================================================================
# CHEMISTRY CALCULATION TOOLS
# ============================================================================

def calculate_molarity(moles: float, volume_liters: float) -> Dict[str, Any]:
    """
    Calculate molarity (concentration) of a solution.

    Molarity (M) = moles of solute / liters of solution

    Parameters:
        moles (float): [Required] Amount of solute in moles
        volume_liters (float): [Required] Volume of solution in liters

    Returns:
        dict: {
            "molarity": float,           # Concentration in mol/L (M)
            "millimolar": float,         # Concentration in mmol/L (mM)
            "classification": str         # "concentrated", "dilute", or "trace"
        }
    """
    if volume_liters <= 0:
        return {
            "error": "Volume must be positive",
            "molarity": 0,
            "millimolar": 0,
            "classification": "invalid"
        }

    molarity = moles / volume_liters
    millimolar = molarity * 1000

    # Classify concentration
    if molarity > 1.0:
        classification = "concentrated"
    elif molarity > 0.001:
        classification = "dilute"
    else:
        classification = "trace"

    return {
        "molarity": round(molarity, 6),
        "millimolar": round(millimolar, 4),
        "classification": classification
    }


def calculate_dilution(initial_molarity: float, initial_volume_ml: float,
                       final_volume_ml: float) -> Dict[str, Any]:
    """
    Calculate final concentration after dilution using C1V1 = C2V2.

    Parameters:
        initial_molarity (float): [Required] Initial concentration in mol/L
        initial_volume_ml (float): [Required] Initial volume in milliliters
        final_volume_ml (float): [Required] Final volume in milliliters

    Returns:
        dict: {
            "final_molarity": float,              # Final concentration in mol/L
            "dilution_factor": float,             # How many times diluted
            "water_to_add_ml": float,             # Volume of water to add
            "instructions": str                   # Step-by-step dilution instructions
        }
    """
    if final_volume_ml < initial_volume_ml:
        return {
            "error": "Final volume must be greater than initial volume for dilution",
            "final_molarity": 0,
            "dilution_factor": 0,
            "water_to_add_ml": 0,
            "instructions": "Invalid dilution"
        }

    # C1V1 = C2V2, so C2 = C1V1/V2
    final_molarity = (initial_molarity * initial_volume_ml) / final_volume_ml
    dilution_factor = final_volume_ml / initial_volume_ml
    water_to_add = final_volume_ml - initial_volume_ml

    instructions = (
        f"1. Measure {initial_volume_ml} mL of {initial_molarity} M solution\n"
        f"2. Add {water_to_add} mL of distilled water\n"
        f"3. Mix thoroughly to obtain {final_volume_ml} mL of {final_molarity:.4f} M solution"
    )

    return {
        "final_molarity": round(final_molarity, 6),
        "dilution_factor": round(dilution_factor, 2),
        "water_to_add_ml": round(water_to_add, 2),
        "instructions": instructions
    }


def calculate_ph(hydrogen_ion_concentration: float) -> Dict[str, Any]:
    """
    Calculate pH from hydrogen ion concentration.

    pH = -log10[H+]

    Parameters:
        hydrogen_ion_concentration (float): [Required] [H+] in mol/L

    Returns:
        dict: {
            "ph": float,                    # pH value
            "poh": float,                   # pOH value (pOH = 14 - pH at 25°C)
            "classification": str,          # "strongly acidic", "acidic", "neutral", "basic", "strongly basic"
            "hydroxide_concentration": float # [OH-] concentration
        }
    """
    if hydrogen_ion_concentration <= 0:
        return {
            "error": "Hydrogen ion concentration must be positive",
            "ph": 0,
            "poh": 0,
            "classification": "invalid",
            "hydroxide_concentration": 0
        }

    ph = -math.log10(hydrogen_ion_concentration)
    poh = 14 - ph
    hydroxide_concentration = 10 ** (-poh)

    # Classify pH
    if ph < 3:
        classification = "strongly acidic"
    elif ph < 7:
        classification = "acidic"
    elif ph == 7:
        classification = "neutral"
    elif ph < 11:
        classification = "basic"
    else:
        classification = "strongly basic"

    return {
        "ph": round(ph, 2),
        "poh": round(poh, 2),
        "classification": classification,
        "hydroxide_concentration": round(hydroxide_concentration, 10)
    }


def calculate_limiting_reagent(reactants: Dict[str, float],
                               stoichiometry: Dict[str, int]) -> Dict[str, Any]:
    """
    Determine the limiting reagent in a chemical reaction.

    Parameters:
        reactants (dict): [Required] Available moles of each reactant, e.g. {"A": 2.0, "B": 3.0}
        stoichiometry (dict): [Required] Stoichiometric coefficients, e.g. {"A": 2, "B": 1, "C": 2}

    Returns:
        dict: {
            "limiting_reagent": str,              # Name of limiting reagent
            "excess_reagents": List[str],         # Names of excess reagents
            "max_product_moles": Dict[str, float], # Maximum moles of each product
            "leftover_reagents": Dict[str, float]  # Moles of excess reagents left over
        }
    """
    # Calculate how many "reaction batches" each reactant can support
    batches_possible = {}
    for reagent, moles_available in reactants.items():
        if reagent in stoichiometry:
            coefficient = stoichiometry[reagent]
            batches_possible[reagent] = moles_available / coefficient

    # The limiting reagent supports the fewest batches
    limiting_reagent = min(batches_possible, key=batches_possible.get)
    max_batches = batches_possible[limiting_reagent]

    # Identify excess reagents
    excess_reagents = [r for r in batches_possible if r != limiting_reagent]

    # Calculate maximum product moles
    max_product_moles = {}
    for species, coefficient in stoichiometry.items():
        if species not in reactants:  # This is a product
            max_product_moles[species] = max_batches * coefficient

    # Calculate leftover reagents
    leftover_reagents = {}
    for reagent in excess_reagents:
        moles_consumed = max_batches * stoichiometry[reagent]
        leftover_reagents[reagent] = reactants[reagent] - moles_consumed

    return {
        "limiting_reagent": limiting_reagent,
        "excess_reagents": excess_reagents,
        "max_product_moles": {k: round(v, 4) for k, v in max_product_moles.items()},
        "leftover_reagents": {k: round(v, 4) for k, v in leftover_reagents.items()}
    }


def ideal_gas_law(pressure_atm: float = None, volume_liters: float = None,
                  moles: float = None, temperature_kelvin: float = None) -> Dict[str, Any]:
    """
    Calculate missing variable using Ideal Gas Law: PV = nRT.

    Provide exactly 3 of the 4 variables, and the function will calculate the 4th.
    R (gas constant) = 0.08206 L·atm/(mol·K)

    Parameters:
        pressure_atm (float): [Optional] Pressure in atmospheres
        volume_liters (float): [Optional] Volume in liters
        moles (float): [Optional] Amount in moles
        temperature_kelvin (float): [Optional] Temperature in Kelvin

    Returns:
        dict: {
            "pressure_atm": float,         # Pressure in atm
            "volume_liters": float,        # Volume in L
            "moles": float,                # Amount in mol
            "temperature_kelvin": float,   # Temperature in K
            "temperature_celsius": float,  # Temperature in °C
            "calculated_variable": str     # Which variable was calculated
        }
    """
    R = 0.08206  # L·atm/(mol·K)

    # Count how many variables are provided
    provided = [pressure_atm, volume_liters, moles, temperature_kelvin]
    non_none_count = sum(x is not None for x in provided)

    if non_none_count != 3:
        return {
            "error": "Must provide exactly 3 variables to calculate the 4th",
            "pressure_atm": 0,
            "volume_liters": 0,
            "moles": 0,
            "temperature_kelvin": 0,
            "temperature_celsius": 0,
            "calculated_variable": "none"
        }

    # Calculate the missing variable
    if pressure_atm is None:
        pressure_atm = (moles * R * temperature_kelvin) / volume_liters
        calculated_variable = "pressure_atm"
    elif volume_liters is None:
        volume_liters = (moles * R * temperature_kelvin) / pressure_atm
        calculated_variable = "volume_liters"
    elif moles is None:
        moles = (pressure_atm * volume_liters) / (R * temperature_kelvin)
        calculated_variable = "moles"
    else:  # temperature_kelvin is None
        temperature_kelvin = (pressure_atm * volume_liters) / (moles * R)
        calculated_variable = "temperature_kelvin"

    temperature_celsius = temperature_kelvin - 273.15

    return {
        "pressure_atm": round(pressure_atm, 4),
        "volume_liters": round(volume_liters, 4),
        "moles": round(moles, 4),
        "temperature_kelvin": round(temperature_kelvin, 2),
        "temperature_celsius": round(temperature_celsius, 2),
        "calculated_variable": calculated_variable
    }


def titration_analysis(acid_volume_ml: float, acid_molarity: float,
                       base_volume_ml: float, base_molarity: float,
                       acid_protons: int = 1, base_hydroxides: int = 1) -> Dict[str, Any]:
    """
    Analyze an acid-base titration to determine equivalence point and excess.

    Parameters:
        acid_volume_ml (float): [Required] Volume of acid in mL
        acid_molarity (float): [Required] Molarity of acid in mol/L
        base_volume_ml (float): [Required] Volume of base added in mL
        base_molarity (float): [Required] Molarity of base in mol/L
        acid_protons (int): [Optional] Number of acidic protons (default=1 for monoprotic)
        base_hydroxides (int): [Optional] Number of hydroxide ions (default=1)

    Returns:
        dict: {
            "acid_moles": float,           # Moles of H+ available
            "base_moles": float,           # Moles of OH- added
            "at_equivalence": bool,        # True if at equivalence point
            "excess_species": str,         # "acid", "base", or "none"
            "excess_moles": float,         # Moles of excess species
            "total_volume_ml": float,      # Total solution volume
            "status": str                  # "under-titrated", "equivalence", "over-titrated"
        }
    """
    # Calculate moles of H+ and OH-
    acid_moles = (acid_volume_ml / 1000) * acid_molarity * acid_protons
    base_moles = (base_volume_ml / 1000) * base_molarity * base_hydroxides

    total_volume_ml = acid_volume_ml + base_volume_ml

    # Determine status
    tolerance = 0.0001  # Small tolerance for equivalence
    if abs(acid_moles - base_moles) < tolerance:
        at_equivalence = True
        excess_species = "none"
        excess_moles = 0
        status = "equivalence"
    elif acid_moles > base_moles:
        at_equivalence = False
        excess_species = "acid"
        excess_moles = acid_moles - base_moles
        status = "under-titrated"
    else:
        at_equivalence = False
        excess_species = "base"
        excess_moles = base_moles - acid_moles
        status = "over-titrated"

    return {
        "acid_moles": round(acid_moles, 6),
        "base_moles": round(base_moles, 6),
        "at_equivalence": at_equivalence,
        "excess_species": excess_species,
        "excess_moles": round(excess_moles, 6),
        "total_volume_ml": round(total_volume_ml, 2),
        "status": status
    }


def check_safety_compatibility(chemical1: str, chemical2: str,
                               safety_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if two chemicals are safe to mix based on safety data.

    Parameters:
        chemical1 (str): [Required] Name of first chemical
        chemical2 (str): [Required] Name of second chemical
        safety_data (dict): [Required] Safety database with incompatibilities

    Returns:
        dict: {
            "safe_to_mix": bool,              # Whether mixing is safe
            "hazard_level": str,              # "none", "low", "medium", "high", "extreme"
            "warnings": List[str],            # List of safety warnings
            "required_ppe": List[str],        # Required personal protective equipment
            "ventilation_required": bool      # Whether fume hood is required
        }
    """
    warnings = []
    required_ppe = ["safety_goggles", "lab_coat", "gloves"]
    ventilation_required = False

    # Check if chemicals are in database
    if chemical1 not in safety_data or chemical2 not in safety_data:
        return {
            "safe_to_mix": False,
            "hazard_level": "unknown",
            "warnings": [f"Unknown chemical: {chemical1 if chemical1 not in safety_data else chemical2}"],
            "required_ppe": required_ppe,
            "ventilation_required": True
        }

    # Get incompatibilities
    incompatible_with_1 = safety_data[chemical1].get("incompatible_with", [])
    incompatible_with_2 = safety_data[chemical2].get("incompatible_with", [])

    # Check for incompatibility
    safe_to_mix = True
    hazard_level = "none"

    if chemical2 in incompatible_with_1 or chemical1 in incompatible_with_2:
        safe_to_mix = False

        # Determine hazard level based on chemical properties
        hazard1 = safety_data[chemical1].get("hazard_class", "low")
        hazard2 = safety_data[chemical2].get("hazard_class", "low")

        hazard_levels_order = ["none", "low", "medium", "high", "extreme"]
        hazard_level = max(hazard1, hazard2, key=lambda x: hazard_levels_order.index(x))

        # Generate warnings
        reaction_type = safety_data[chemical1].get("reaction_with", {}).get(chemical2, "violent reaction")
        warnings.append(f"DANGER: {chemical1} and {chemical2} are incompatible!")
        warnings.append(f"Potential hazard: {reaction_type}")

        # Additional PPE for dangerous combinations
        if hazard_level in ["high", "extreme"]:
            required_ppe.extend(["face_shield", "apron", "fume_hood"])
            ventilation_required = True
            warnings.append("CRITICAL: Use fume hood and additional protective equipment")

    # Check if either chemical requires ventilation
    if safety_data[chemical1].get("volatile", False) or safety_data[chemical2].get("volatile", False):
        ventilation_required = True
        required_ppe.append("fume_hood")

    return {
        "safe_to_mix": safe_to_mix,
        "hazard_level": hazard_level,
        "warnings": warnings,
        "required_ppe": list(set(required_ppe)),  # Remove duplicates
        "ventilation_required": ventilation_required
    }


# ============================================================================
# LABORATORY VARIABLES
# ============================================================================

# Chemical inventory with properties
chemical_inventory = {
    "HCl": {"name": "Hydrochloric Acid", "molarity": 6.0, "volume_ml": 500, "hazard": "corrosive"},
    "NaOH": {"name": "Sodium Hydroxide", "molarity": 3.0, "volume_ml": 500, "hazard": "corrosive"},
    "H2SO4": {"name": "Sulfuric Acid", "molarity": 2.0, "volume_ml": 250, "hazard": "corrosive"},
    "NaCl": {"name": "Sodium Chloride", "molarity": 1.0, "volume_ml": 1000, "hazard": "none"},
    "AgNO3": {"name": "Silver Nitrate", "molarity": 0.1, "volume_ml": 100, "hazard": "toxic"},
}

# Safety data
safety_database = {
    "HCl": {
        "hazard_class": "high",
        "incompatible_with": ["NaOH", "bleach"],
        "volatile": True,
        "reaction_with": {
            "NaOH": "exothermic neutralization (releases heat)",
            "bleach": "releases toxic chlorine gas"
        }
    },
    "NaOH": {
        "hazard_class": "high",
        "incompatible_with": ["HCl", "H2SO4", "acids"],
        "volatile": False,
        "reaction_with": {
            "HCl": "exothermic neutralization (releases heat)",
            "H2SO4": "violent exothermic reaction"
        }
    },
    "H2SO4": {
        "hazard_class": "extreme",
        "incompatible_with": ["NaOH", "water_rapid", "bases"],
        "volatile": False,
        "reaction_with": {
            "NaOH": "violent exothermic reaction",
            "water_rapid": "explosive boiling and spattering"
        }
    },
    "NaCl": {
        "hazard_class": "low",
        "incompatible_with": [],
        "volatile": False
    },
    "AgNO3": {
        "hazard_class": "medium",
        "incompatible_with": ["NaCl"],
        "volatile": False,
        "reaction_with": {
            "NaCl": "forms white AgCl precipitate"
        }
    }
}

# Lab equipment
lab_equipment = {
    "graduated_cylinders": [10, 25, 50, 100, 250, 500],  # mL
    "beakers": [50, 100, 250, 500, 1000],  # mL
    "pipettes": [1, 5, 10, 25],  # mL
    "burette": {"volume_ml": 50, "precision": 0.05},
    "ph_meter": {"precision": 0.01, "calibrated": True},
    "balance": {"max_weight_g": 200, "precision": 0.001},
    "fume_hood": {"available": True}
}


# ============================================================================
# VALIDATORS
# ============================================================================

def validate_dilution_calculation(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate dilution calculation accuracy."""
    try:
        final_molarity = runtime.get_variable_value("final_molarity")
        water_needed = runtime.get_variable_value("water_needed_ml")

        # Expected values (0.5 M HCl diluted from 6.0 M, 100 mL final volume)
        # C1V1 = C2V2 → V1 = C2V2/C1 = (0.5 * 100) / 6.0 = 8.33 mL
        # Water = 100 - 8.33 = 91.67 mL
        expected_molarity = 0.5
        expected_water = 91.67

        # Check with tolerance
        molarity_correct = abs(final_molarity - expected_molarity) < 0.01
        water_correct = abs(water_needed - expected_water) < 0.5

        if molarity_correct and water_correct:
            return ValidatorResult(True, "Dilution calculated correctly!")
        else:
            errors = []
            if not molarity_correct:
                errors.append(f"Molarity incorrect: got {final_molarity}, expected {expected_molarity}")
            if not water_correct:
                errors.append(f"Water volume incorrect: got {water_needed} mL, expected {expected_water} mL")
            return ValidatorResult(False, "; ".join(errors))

    except Exception as e:
        return ValidatorResult(False, f"Error checking variables: {str(e)}")


def validate_ph_calculation(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate pH calculation."""
    try:
        ph_value = runtime.get_variable_value("solution_ph")
        classification = runtime.get_variable_value("ph_classification")

        # Expected: pH of 0.01 M HCl = -log10(0.01) = 2.0 (strongly acidic)
        expected_ph = 2.0
        expected_class = "strongly acidic"

        ph_correct = abs(ph_value - expected_ph) < 0.1
        class_correct = classification == expected_class

        if ph_correct and class_correct:
            return ValidatorResult(True, "pH calculation correct!")
        else:
            return ValidatorResult(False,
                f"pH: got {ph_value} (expected {expected_ph}), "
                f"classification: got {classification} (expected {expected_class})")

    except Exception as e:
        return ValidatorResult(False, f"Error: {str(e)}")


def validate_limiting_reagent(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate limiting reagent identification."""
    try:
        limiting = runtime.get_variable_value("limiting_reagent")
        max_product = runtime.get_variable_value("max_AgCl_moles")

        # Expected: AgNO3 + NaCl → AgCl + NaNO3
        # 0.01 mol AgNO3, 0.02 mol NaCl → AgNO3 is limiting
        # Max AgCl = 0.01 mol
        expected_limiting = "AgNO3"
        expected_product = 0.01

        limiting_correct = limiting == expected_limiting
        product_correct = abs(max_product - expected_product) < 0.001

        if limiting_correct and product_correct:
            return ValidatorResult(True, "Limiting reagent correctly identified!")
        else:
            return ValidatorResult(False,
                f"Limiting reagent: got {limiting} (expected {expected_limiting}), "
                f"max product: got {max_product} mol (expected {expected_product} mol)")

    except Exception as e:
        return ValidatorResult(False, f"Error: {str(e)}")


def validate_gas_law(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate ideal gas law calculation."""
    try:
        pressure = runtime.get_variable_value("gas_pressure_atm")

        # Expected: PV = nRT → P = nRT/V
        # n=2.0 mol, T=298 K, V=10 L, R=0.08206
        # P = (2.0 * 0.08206 * 298) / 10 = 4.89 atm
        expected_pressure = 4.89

        pressure_correct = abs(pressure - expected_pressure) < 0.1

        if pressure_correct:
            return ValidatorResult(True, "Gas law calculation correct!")
        else:
            return ValidatorResult(False,
                f"Pressure: got {pressure} atm (expected {expected_pressure} atm)")

    except Exception as e:
        return ValidatorResult(False, f"Error: {str(e)}")


def validate_titration(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate titration analysis."""
    try:
        equivalence = runtime.get_variable_value("at_equivalence_point")
        status = runtime.get_variable_value("titration_status")

        # Expected: 25 mL of 0.1 M HCl + 25 mL of 0.1 M NaOH
        # Moles equal → at equivalence
        expected_equivalence = True
        expected_status = "equivalence"

        equiv_correct = equivalence == expected_equivalence
        status_correct = status == expected_status

        if equiv_correct and status_correct:
            return ValidatorResult(True, "Titration analysis correct!")
        else:
            return ValidatorResult(False,
                f"At equivalence: got {equivalence} (expected {expected_equivalence}), "
                f"status: got {status} (expected {expected_status})")

    except Exception as e:
        return ValidatorResult(False, f"Error: {str(e)}")


def validate_safety_check(
    response: str,
    runtime: PythonRuntime,
    turn: BenchmarkTurn,
    actual_calls: List[ToolCall]
) -> ValidatorResult:
    """Validate safety compatibility check."""
    try:
        safe = runtime.get_variable_value("safe_to_mix")
        hazard = runtime.get_variable_value("hazard_level")

        # Expected: HCl + NaOH → NOT safe (exothermic), high hazard
        expected_safe = False
        expected_hazard = "high"

        safe_correct = safe == expected_safe
        hazard_correct = hazard == expected_hazard

        if safe_correct and hazard_correct:
            return ValidatorResult(True, "Safety check correct!")
        else:
            return ValidatorResult(False,
                f"Safe to mix: got {safe} (expected {expected_safe}), "
                f"hazard level: got {hazard} (expected {expected_hazard})")

    except Exception as e:
        return ValidatorResult(False, f"Error: {str(e)}")


# ============================================================================
# EXPORTS
# ============================================================================

tools = [
    calculate_molarity,
    calculate_dilution,
    calculate_ph,
    calculate_limiting_reagent,
    ideal_gas_law,
    titration_analysis,
    check_safety_compatibility
]

variables = [
    Variable("chemical_inventory", chemical_inventory, "Available chemicals with molarity and volume"),
    Variable("safety_database", safety_database, "Chemical safety and compatibility database"),
    Variable("lab_equipment", lab_equipment, "Available laboratory equipment")
]

validators = {
    "validate_dilution": validate_dilution_calculation,
    "validate_ph": validate_ph_calculation,
    "validate_limiting_reagent": validate_limiting_reagent,
    "validate_gas_law": validate_gas_law,
    "validate_titration": validate_titration,
    "validate_safety": validate_safety_check
}
