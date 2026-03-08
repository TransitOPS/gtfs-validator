1. Compared files: old = `out/report-original.json` (Java validator 7.1.0, validated at 2026-03-08T10:24:30-04:00), new = `out/report-new.json` (Python validator 0.1.0, validated at 2026-03-08T15:13:35.003357+00:00). ❌
2. Report size differs: old = 113,340 bytes, new = 179,455 bytes. ❌
3. Runtime differs: old `validationTimeSeconds` = 4.215713167, new `validationTimeSeconds` = 66.115. ❌
4. Country code differs: old `countryCode` = `US`, new `countryCode` = `ZZ`.
5. GTFS input path format differs: old `gtfsInput` = `file:///Users/ryanmahoney/Documents/gtfs-validator/out/gtfs-mbta.zip`, new `gtfsInput` = `out/gtfs-mbta.zip`.
6. Output report filenames differ in summary metadata: old points to `report.json`/`system_errors.json`/`report.html`, new points to `report-new.json`/`system_errors-new.json`/`report-new.html`.
7. Total notice count differs: old = 859, new = 871.
8. Notice group count differs: old = 12 groups, new = 16 groups.
9. `summary.counts.Shapes` differs: old = 1119 (distinct shape IDs), new = 384403 (appears to be raw shape row count).
10. `summary.feedInfo` formatting differs: old uses normalized values (`feedLanguage` = `English`, ISO-like dates), new uses raw GTFS values (`feedLanguage` = `EN`, dates like `20260227`).
11. `summary.agencies` content differs: old includes extra keys (`email`, `timezone`), new includes only `name`, `url`, `phone` for this feed.
12. `summary.files` differs in composition: old includes unknown extension files from the feed (e.g., `calendar_attributes.txt`, `route_patterns.txt`), new includes extensionless aliases plus `.txt` keys (e.g., both `calendar` and `calendar.txt`), increasing file entries from 32 (old) to 40 (new).
13. Feature naming mismatch exists: old includes `In-station Traversal Time`, new includes `Traversal Time` (same concept, different label).
14. Notice code naming mismatch exists for feed expiration: old emits `feed_expiration_date30_days`, new emits `feed_expiration_date_30_days` (underscore insertion).
15. Notice code count mismatch exists for `stop_too_far_from_shape`: old = 13, new = 12.
16. Notice groups present only in old: `feed_expiration_date30_days` (1) and old-count version of `stop_too_far_from_shape` (13).
17. Notice groups present only in new: `big_gap_in_service` (5), `feed_expiration_date_30_days` (1), `leading_or_trailing_whitespaces` (3), `mixed_case_recommended_field` (4), `service_has_no_active_day_of_the_week` (1), and new-count version of `stop_too_far_from_shape` (12).
18. Notice groups with matching counts old vs new: `expired_calendar` (8), `fast_travel_between_consecutive_stops` (65), `fast_travel_between_far_stops` (33), `platform_without_parent_station` (3), `route_short_name_too_long` (213), `same_name_and_description_for_stop` (324), `stop_without_stop_time` (151), `unknown_column` (29), `unknown_file` (12), and `unusable_trip` (7).
19. System errors context for latest successful new run: `out/system_errors-new.json` is empty (`{"notices": []}`), so differences above are output-contract/content differences rather than runtime crashes in that run.
