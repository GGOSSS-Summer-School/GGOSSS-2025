#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
import copernicusmarine  
import warnings
warnings.filterwarnings('ignore') 

# Définir les dates de début et de fin
start_date = datetime(2020, 11, 26)
end_date = datetime(2020, 12, 31)

# Définir les variables à télécharger
variables = ["uo","vo", "thetao", "so", "mlotst", "zos"]

# Boucle sur le temps
current_date = start_date
while current_date <= end_date:
    # Formatage de la date de début pour le jour courant
    start_datetime = current_date.strftime("%Y-%m-%d")
    
    # Construire le nom du fichier de sortie basé sur la date actuelle
    output_filename = f"GG_cmems_data_{current_date.strftime('%Y%m%d')}.nc"
    
    try:
        # Appeler la fonction copernicus pour télécharger les variables spécifiées
        copernicusmarine.subset(
            dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
            variables=variables,
            minimum_longitude=-10.0,
            maximum_longitude=12.0,
            minimum_latitude=-6,
            maximum_latitude=7,
            start_datetime=start_datetime,
            end_datetime=start_datetime,  # Utilisez la même date ici
            minimum_depth=0,
            maximum_depth=50,
            output_filename=output_filename,
            output_directory="/home/user/data_copernicus/",
            username="**********",
            password="**********"
        )
        print(f"Téléchargement des données pour {start_datetime} réussi.\n")
    except Exception as e:
        print(f"Échec du téléchargement des données pour {start_datetime} : {e}")
    
    # Incrémenter la date courante d'un jour
    current_date += timedelta(days=1)

print("\n\nTous les téléchargements sont terminés !")

exit()

