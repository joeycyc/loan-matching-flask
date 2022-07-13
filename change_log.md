# Change Log

## Versioning convention
1. In format of [year].[month].[day][a/b/...]
2. a = stable release, b = Beta version

## Versions
### v22.6.17a
1. Adopted modular pattern to decouple data loading, preprocessing, matching and visualization
2. Matching algorithm = largest overlapping area + by 4 stages (incl. initial stage)
3. Visualization = get_gantt_2 where consisting 3 subplots
    - project shortfall + matched,
    - leftover
    - equity

### v22.6.26a
1. Added matching scheme #2 for replacing Committed Revolver with Uncommitted Revolver
2. Different bar height [under testing]

### v22.6.29a
1. Uncommitted Revolver is evergreen (no expiry date)

### v22.7.4a
1. Added parameters for tuning the Uncommitted Revolver replacement criteria: 
    1. Whether UC-RTN has to fully cover matched C-RTN entry
    2. Whether the saving is calculated by net_margin difference x overlapping area or 
    net_margin difference only
2. Reviewed tooltip
3. Added "Total shortfall"

### v22.7.6b
1. Added new chart type with different bar height depending on amount
2. Added Matching parameters selection panel
3. Added Current Matching Scheme
4. Combined callback functions
5. Added Buttons and Slide Bars

### v22.7.11b
1. Enhanced UI
2. Combined Rerun matching and refresh dashboard
3. Added button to toggle matching scheme

### v22.7.13b
1. Integrated Dash into Flask, to make it a web application
