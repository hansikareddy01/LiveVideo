"""Approximate coordinates (latitude, longitude) of the destinations, for the India map."""
_C = {
 "Jaipur": (26.91, 75.79), "Udaipur": (24.58, 73.68), "Agra": (27.18, 78.01), "Mumbai": (19.08, 72.88),
 "Kolkata": (22.57, 88.36), "Chennai": (13.08, 80.27), "Bengaluru": (12.97, 77.59), "Hyderabad": (17.39, 78.49),
 "New Delhi": (28.61, 77.21), "Pune": (18.52, 73.86), "Amritsar": (31.63, 74.87), "Shimla": (31.10, 77.17),
 "Manali": (32.24, 77.19), "McLeod Ganj": (32.24, 76.32), "Rishikesh": (30.09, 78.27), "Mussoorie": (30.46, 78.07),
 "Nainital": (29.38, 79.46), "Leh": (34.15, 77.58), "Srinagar": (34.08, 74.80), "Gulmarg": (34.05, 74.38),
 "Varanasi": (25.32, 82.97), "Jodhpur": (26.24, 73.02), "Jaisalmer": (26.92, 70.90), "Pushkar": (26.49, 74.55),
 "Mount Abu": (24.59, 72.71), "Khajuraho": (24.85, 79.92), "Darjeeling": (27.04, 88.26), "Gangtok": (27.33, 88.61),
 "Shillong": (25.57, 91.88), "Puri": (19.81, 85.83), "Ahmedabad": (23.02, 72.57), "Lonavala": (18.75, 73.41),
 "Mahabaleshwar": (17.92, 73.66), "Munnar": (10.09, 77.06), "Alappuzha": (9.49, 76.33), "Thekkady": (9.60, 77.17),
 "Kovalam": (8.40, 76.98), "Ooty": (11.41, 76.69), "Kodaikanal": (10.24, 77.49), "Madikeri": (12.42, 75.74),
 "Hampi": (15.34, 76.46), "Pondicherry": (11.94, 79.81), "Madurai": (9.92, 78.12), "Rameswaram": (9.29, 79.31),
 "Kanyakumari": (8.08, 77.54), "Goa": (15.50, 73.83), "Kochi": (9.93, 76.27), "Ranthambore": (26.02, 76.50),
 "Mysuru": (12.30, 76.64), "Varkala": (8.73, 76.72), "Andaman Islands": (11.62, 92.73),
}
COORDS = {k.title(): v for k, v in _C.items()}   # data_prep title-cases names, e.g. "Mcleod Ganj"
