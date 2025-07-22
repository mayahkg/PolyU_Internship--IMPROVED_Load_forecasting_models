from models.HKisland_models.dataloader import read_data_for_hkisland_model
from config import building_name_list

if __name__ == '__main__':
    # read data for all buildings
    for building_name in building_name_list:
        read_data_for_hkisland_model(building_name=building_name,
                                     temperature_min=0,
                                     temperature_max=40)

    # read data for specified set of buildings
    # for building_name in ['OIE']:
    #     read_data_for_hkisland_model(building_name=building_name,
    #                                  temperature_min=0,
    #                                  temperature_max=40)
