# ONE WAY PICKZ — Tennis V4 Streamlit

GitHub / Streamlit-ready tennis projection app.

## What V4 adds
- Underdog tennis line pull with manual/upload fallback
- ATP/WTA historical match-stat engine
- Aces, games won, total games, breaks, break points, double faults, fantasy points
- Surface, indoor/outdoor, tournament level, best-of-3 / best-of-5
- Serving / returning / rally / workload metrics
- True metric overlay support: winners, unforced errors, forced errors, rally length, distance/workload
- Injury / retirement flag overlay
- Draw / match status overlay
- Tennis learning engine: player + prop + surface + tournament bias
- **CLV tracker**: upload closing lines and learn if picks beat the close
- **Sample-size gates**: blocks weak samples from becoming official plays
- Remote CSV import: paste raw CSV URLs for free charting/status/draw/CLV sheets

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Cloud
1. Upload all files to a GitHub repo.
2. Go to Streamlit Cloud.
3. Select the repo.
4. Main file path: `streamlit_app.py`.
5. Deploy.

## CSV templates
- `samples/manual_tennis_lines_template.csv` — manual Underdog lines
- `sample_charting_true_metrics.csv` — true charting overlay
- `sample_status_flags.csv` — injury/retirement flags
- `sample_draw_status.csv` — draw/match status
- `sample_closing_lines.csv` — closing lines for CLV tracking

## Notes
Underdog endpoints can change or block access. The app includes manual/upload fallback so projections still work.

Free tennis data does not consistently provide official live injury feeds or every match's unforced errors/winners/rally length. V4 handles this with overlays and remote CSV imports, which is the most realistic free approach.
