# System Errors

All errors are `runtime_exception_in_validator` (26 total).

1. [x] **block_trips_overlapping** — `ColumnNotFoundError`: unable to find column `csv_row_number`
2. [x] **trip_usability** — `KeyError`: `csv_row_number`
3. [x] **trip_and_shape_distance** — `ColumnNotFoundError`: unable to find column `shape_dist_traveled`
4. [x] **service_gap** — `AttributeError`: `'str' object has no attribute 'weekday'`
5. [x] **service_no_active_day** — `ColumnNotFoundError`: unable to find column `csv_row_number`
6. [x] **feed_contact** — `ColumnNotFoundError`: unable to find column `feed_contact_url`
7. [x] **feed_expiration_date** — `KeyError`: `csv_row_number`
8. [x] **feed_valid_today** — `TypeError`: `'>' not supported between instances of 'str' and 'datetime.date'`
9. [x] **location_has_stop_times** — `KeyError`: `csvRowNumber`
10. [x] **location_type_single_entity** — `KeyError`: `csvRowNumber`
11. [x] **missing_level_id** — `ColumnNotFoundError`: unable to find column `csvRowNumber`
12. [x] **parent_station** — `ColumnNotFoundError`: unable to find column `csvRowNumber`
13. [x] **stop_name** — `KeyError`: `csv_row_number`
14. [x] **pathway_endpoint_type** — `ColumnNotFoundError`: unable to find column `csv_row_number`
15. [x] **pathway_loop** — `ColumnNotFoundError`: unable to find column `csv_row_number`
16. [x] **route_name** — `KeyError`: `csv_row_number`
17. [x] **shape_increasing_distance** — `ColumnNotFoundError`: unable to find column `csv_row_number`
18. [x] **shape_to_stop_matching** — `ColumnNotFoundError`: unable to find column `csv_row_number`
19. [x] **shape_usage** — `ColumnNotFoundError`: unable to find column `csv_row_number`
20. [x] **single_shape_point** — `ColumnNotFoundError`: unable to find column `csv_row_number`
21. [x] **stop_time_travel_speed** — `ColumnNotFoundError`: unable to find column `csv_row_number`
22. [x] **stop_times_geography_id_presence** — `ColumnNotFoundError`: unable to find column `csv_row_number`
23. [x] **stop_times_trip_block_order** — `ColumnNotFoundError`: unable to find column `csv_row_number`
24. [x] **timeframe_start_and_end_time** — `ColumnNotFoundError`: unable to find column `csv_row_number`
25. [x] **unique_geography_id** — `ColumnNotFoundError`: unable to find column `csv_row_number`
26. [x] **url_consistency** — `ColumnNotFoundError`: unable to find column `csv_row_number`
