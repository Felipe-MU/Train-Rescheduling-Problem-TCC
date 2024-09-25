from Supportingcode.Classes import Train, track, station
# Data reader
class Track_name:
    def __init__(self, content) -> None:
        self.content = content
        if not self.content.isdigit():
            d = self.content.split('-')
            self.station = '-'.join(d[1:len(d)])
        else:
            self.station = False

    def Find_number(self):
        if self.content.isdigit():
            return int(self.content)
        else:
            return int(self.content.split('-')[0])
        

class Data:
    def __init__(self, name, local, document) -> None:
        self.name = name
        self.local = local
        self.document = document
    
    def read_data_edb(self):
        tracks = {}
        stations = {}
        trains = {}
        with open(f'{self.local}{self.name}.edb', 'r') as file:
            for line in file:
                line = line.replace('.', ' ').replace('"', " ")
                line = line.replace(',',' ').replace('(', ' ').replace(')', ' ').split()
                if line != []:
                    # getting tracks in the data
                    if line[0] == 'track':
                        T = Track_name(line[1])
                        tracks[T.Find_number()] = track(T.Find_number())
                        # tracks[T.Find_number()].station = T.station
                    if line[0] == 'track_next':
                        T1 = Track_name(line[1])
                        T2 = Track_name(line[2])
                        tracks[T1.Find_number()].add_next(T2.Find_number())
                    # getting stations
                    if line[0] == 'station':
                        stations[line[1]] = station(line[1])
                    if line[0] == 'station_tracks':
                        T = Track_name(line[2])
                        stations[line[1]].add_track(T.Find_number())
                    # getting trains
                    if line[0] =='train':
                        if int(line[1][1:]) not in trains.keys():
                            trains[int(line[1][1:])] = Train(int(line[1][1:]))
                    if line[0] == 'train_start':
                        trains[int(line[1][1:])].start = line[2]
                    if line[0] == 'train_destination':
                        trains[int(line[1][1:])].destination = line[2]
                    if line[0] == 'route_direction':
                        trains[int(line[1][1:])].direction = line[2]
                    if line[0] == 'route_first':
                        T = Track_name(line[2])
                        trains[int(line[1][1:])].defining_route(0, T.Find_number())
                    if line[0] == 'route_next':
                        T1 = Track_name(line[2])
                        T2 = Track_name(line[3])
                        trains[int(line[1][1:])].defining_route(T1.Find_number(), T2.Find_number())
                        # if T1.station:
                        #     trains[int(line[1][1:])].stations[T1.Find_number()] = T1.station
                        # if T2.station:
                        #     trains[int(line[1][1:])].stations[T2.Find_number()] = T2.station
                    
                    if line[0] == 'train_stay':
                        trains[int(line[1][1:])].planned_stop[line[2]] = int(line[3])

                    if line[0] == 'train_timeontrack':
                        T = Track_name(line[2])
                        trains[int(line[1][1:])].timeontrack[T.Find_number()] = int(line[3])

                    if line[0] == 'current_schedule_begin':
                        if int(line[1][1:]) not in trains.keys():
                            trains[int(line[1][1:])] = Train(int(line[1][1:]))
                        T = Track_name(line[2])
                        trains[int(line[1][1:])].defining_current_schedule(T.Find_number(), True, int(line[3]))

                    if line[0] == 'current_schedule_end':
                        T= Track_name(line[2])
                        trains[int(line[1][1:])].defining_current_schedule(T.Find_number(), False, int(line[3]))

                    if line[0] == 'current_time':
                        current_time = int(line[1])
                    
                    if line[0] == 'maxTime':
                        maxtime = int(line[1])


        Train.define_dummhnode_timeontrack(trains)
        [train.defining_stations(stations) for train in trains.values()]
        return (tracks, stations, trains, current_time, maxtime)
    
    def read_data_xml(self):
        return


    def write_to_CSV(self):
        import pandas as pd
        dfs ={}
        if self.document == "edb":
            tracks, stations, trains, current_time, maxtime = self.read_data_edb()
        else:
            tracks, stations, trains, current_time, maxtime = self.read_data_xml()
        dfs['Train']={ 'current_route':{train.id:train.current_route for train in trains.values()}, 
        'begin_schedule':{train.id:{trackk: train.begin_schedule[trackk] for trackk in train.begin_schedule} for train in trains.values() }, 
        'end_schedule':{train.id:{trackk: train.end_schedule[trackk] for trackk in train.end_schedule} for train in trains.values() },
        'timeontrack': {train.id:{trackk: train.timeontrack[trackk] for trackk in train.timeontrack} for train in trains.values() },
        'planned_stop':{train.id:{trackk: train.planned_stop[trackk] for trackk in train.planned_stop} for train in trains.values() },
        'stations':{train.id:{trainstations: train.stations[trainstations] for trainstations in train.stations} for train in trains.values() }}
        
        dfs['track']={'next':{Track.id: Track.next for Track in tracks.values() }}
        
        dfs['station']={ 'tracks':{Station.name: Station.tracks for Station in stations.values()}}
        
        
        dfs['rest']={ 'maxtime': [maxtime],
                         'current_time': [current_time]}
        

        import os
        name = self.name.split('.')[0]
        for df in dfs:
            new_dir = f'CSVFiles/{name}/{df}'
            path = os.path.join(f'{self.local}', new_dir)
            os.makedirs(path)
            for k in dfs[df]:
                data = pd.DataFrame({k:dfs[df][k]}).to_csv(f'{self.local}CSVFiles/{name}/{df}/{k}.csv')
        
            
    

    def read_from_CSV(self):
        import pandas as pd
        import os
        name = self.name.split('.')[0]
        attributes = {}
        data = {}
        trains = {}
        tracks = {}
        stations = {}
        for objects in os.listdir(f'{self.local}CSVFiles/{name}'):
            attributes[objects] = [atribute for atribute in os.listdir(f'{self.local}CSVFiles/{name}/{objects}')]
        dfs = {}
        for classe in attributes:
            dfs[classe] = []
            for attribute in attributes[classe]:
                df = pd.read_csv(f'{self.local}CSVFiles/{name}/{classe}/{attribute}')
                if classe == 'stations':
                    df.rename(columns = {'Unnamed: 0':'name'}, inplace = True)
                else:
                    df.rename(columns = {'Unnamed: 0':'id'}, inplace = True)
                dfs[classe].append(df.to_dict())
        for classe in dfs:
            if classe != 'rest':
                data[classe] = {}
                for dictionary in dfs[classe]:
                    id, attribute_name = dictionary.keys()
                    for objectid in dictionary[id].values():
                        if not objectid in data[classe]:
                            data[classe][objectid] = eval(classe+'('+"'"+f'{objectid}'+"'"+')')
                        setattr(data[classe][objectid], attribute_name, eval(dictionary[attribute_name][list(dictionary[id].values()).index(objectid)]))
            else:
                for dictionary in dfs[classe]:
                    attribute_name = list(dictionary.keys())[1]
                    data[attribute_name] = dictionary[attribute_name][0]
        current_time, maxtime, stations, tracks, trains = data.values() 
        Train.define_dummhnode_timeontrack(trains) 
        for train in trains.values():
            train.defineST(True)
        return tracks, stations, trains, current_time, maxtime