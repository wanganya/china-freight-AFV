# Highway Flow Observation Data

This directory contains highway traffic-flow observations at two temporal resolutions.

The manuscript describes observations from 2,569 highways covering approximately 258,160 km, with an overall sampling rate of 4.8%. Traffic data were collected using microwave or geomagnetic vehicle detectors.

Two temporal datasets are used:

1. monthly traffic observations for January-December 2019;
2. hourly observations for one representative week in each season.

The four seasonal weeks used by the code are:

- Winter: 2019-01-14 to 2019-01-20
- Spring: 2019-04-18 to 2019-04-24
- Summer: 2019-07-16 to 2019-07-22
- Autumn: 2019-11-06 to 2019-11-12

## Structure

```text
traffic_volume/
├── README.md
├── monthly/
│   └── MonthDistribution.txt
└── hourly/
    ├── 20190114_update.xls
    ├── ...
    └── 20191112_update.xls
```

Stage 2 uses both monthly and hourly data. Stages 3 and 4 use the hourly traffic-condition files.

See the READMEs in `/monthly` and `/hourly` for details.
