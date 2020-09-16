import numpy as np


def sphere_fibo(nb_points):
    """
    Entrée le nombre de points de la spère
    créé des points uniformement repartis sur la sphère de centre (0,0,0)
    Renvoie un numpy.array des points

    """
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(nb_points)
    z = np.linspace(1 - 1.0 / nb_points, 1.0 / nb_points - 1, nb_points)
    radius = np.sqrt(1 - z * z)

    points = np.zeros((nb_points, 3))
    points[:, 0] = radius * np.cos(theta)
    points[:, 1] = radius * np.sin(theta)
    points[:, 2] = z

    return points
