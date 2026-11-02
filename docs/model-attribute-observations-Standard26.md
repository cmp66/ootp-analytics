## What the ranking says (written by hand for the Model-Sim-26-Std run of 2026-11-02; check it after retraining)

The OOTP 26 ranking follows the OOTP 27 one (`model-attribute-ranking.md`) in most places. The differences are listed last.

- **Batting:** power (53 % of the model's skill) and avoiding strikeouts (29 %) carry most of the value; eye is 10 %, BABIP
  7 % and gap power 1 %. +10 points of power is worth about 11.1 runs per 600 PA, strikeout avoidance 7.1, BABIP 6.9,
  eye 5.4 and gap 2.5.
- **Baserunning:** stealing (`STE`) is 81 % of it (+10 points: 1.7 runs per 600 PA); baserunning (`RUN`) and speed
  (`SPE`) are 12 % and 8 % (0.3 to 0.4 runs per +10 points).
- **Pitching:** home runs allowed (47 %) leads, then stuff (33 %) and control (11 %): +10 points are worth 8.3, 5.6 and
  3.9 runs per 200 IP. PBABIP acts through the BABIP part, where +10 points is worth about 2.7 runs per 200 IP. Pitch
  count (-1.0 runs per pitch) and stamina (-0.3 per +10 points) are small negative effects; as in OOTP 27 they most likely
  mark the pitcher's role (value is measured against his own usage group) rather than penalise him, but this has not
  been checked per usage group for OOTP 26.
- **Fielding:** range is nearly everything at the outfield positions, 2B and 1B (91 % to 100 %). Infield arm matters most
  at third base (58 % against 39 % for range), then shortstop (21 %), little at second (6 %); outfield arm is worth
  about 1 % or less; error rating 0 % to 3 % for infielders. Catcher framing is 90 % of the catcher's value, blocking
  (`C ABI`) the rest. First basemen get 9 % of their model from height.
  +10 points of range: CF 17.2, SS 14.2, 2B 11.8, LF 5.9, RF 4.5, 3B 2.7, 1B 2.5 runs per season.

**Compared with OOTP 27**
- Batting ratings are worth more per point in OOTP 26 (power 11.1 against 9.6, BABIP 6.9 against 5.4, eye 5.4 against
  3.5 runs per +10 points).
- Baserunning: stealing carries more (81 % against a share of 92 % for `RUN` and `STE` together in OOTP 27, where `RUN`
  and `STE` were of similar size); `RUN` is now small (12 %).
- Pitching: stuff and home runs allowed are worth more per point (5.6 and 8.3 against 4.9 and 6.5 runs per +10
  points), control is a little more (3.9 against 3.3).
- Fielding: range at third base is worth half as much per point (2.7 against 5.4 runs), at left field twice as much (5.9
  against 2.8), at center field and shortstop about a quarter more. The 3B, 1B and C models fit worse in OOTP 26 (R² 0.35,
  0.39 and 0.58 against 0.61, 0.37 and 0.85) and the catcher model depends a little less on framing.
