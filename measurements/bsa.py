

def haycock_BSA(height, weight):
    """
    Calculate BSA (height: cm, weight: kg)
    """
    BSA = 0.024265 * (height ** 0.3964) * (weight ** 0.5378)

    return BSA

def DuBois_BSA(height, weight):
    """
    Calculate BSA (height: cm, weight: kg)
    """
    BSA = 0.20247 * ((height/100) ** 0.725) * (weight ** 0.425)

    return BSA
