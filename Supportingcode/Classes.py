# Basic Classes to the problem
import pandas as pd
# trains
class Train:
    def __init__(self, id) -> None:
        self.id = int(id)
        self.current_route = [0,94]
        self.begin_schedule = {}
        self.end_schedule = {}
        self.timeontrack = {}
        self.planned_stop = {}


    def defining_route(self, previous_track, next_track):
        if previous_track in self.current_route:
            local = self.current_route.index(previous_track)
            self.current_route.insert(local+1, next_track)
    
    # def add_buffer_to_route(self):
    #     while len(self.current_route_buffer)>0:
    #         for k in range(len(self.current_route_buffer)):
    #             if self.current_route_buffer[k][0] in self.current_route:
    #                 local = self.current_route.index(self.current_route_buffer[k][0])
    #                 self.current_route.insert(local+1, self.current_route_buffer[k][1])
    #                 self.current_route.pop(k)
    #                 print(self.current_route_buffer)


    def add_last_track(self):
        self.current_route.append(self.add_last_track)
    
    def defining_current_schedule(self, track, begin_end, time):
        if begin_end:
            self.begin_schedule[track] = time
        else:
            self.end_schedule[track] = time

    # def setting_time_spent(self, type, track, timespent):
    #     if type is "Running":
    #         self.timeontrack[track]= timespent
    #     elif type is "Stop":
    #         self.planned_stop[track]= timespent
    def defining_stations(self, stations):
        self.stations = {}
        for station in stations.values():
            for tracks in set(self.current_route).intersection([tracks for tracks in station.tracks]):
                self.stations[tracks] =  station.name 

    
    def __repr__(self) -> str:
        return f'{self.timeontrack}'
    

    # @classmethod
    @staticmethod
    def define_dummhnode_timeontrack(trains):
        for train in trains.values():
            train.timeontrack[0] = train.begin_schedule[train.current_route[1]]

    
# Rail system

class track:
    def __init__(self, id) -> None:
        self.id = int(id)
        self.next = []
        
    def add_next(self, next_track):
        if next_track not in self.next:
            self.next.append(next_track)

    def __repr__(self) -> str:
        return f"track:{self.id} conections: {self.next}"


class station:
    def __init__(self, name) -> None:
        self.name = name
        self.tracks = []
        
    def add_track(self, track):
        self.tracks.append(track)

    def __repr__(self) -> str:
        return f'Station: {self.name} tracks: {self.tracks}'