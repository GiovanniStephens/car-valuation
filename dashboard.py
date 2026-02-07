#!/usr/bin/env python3
"""
Streamlit dashboard for car valuation models.

Launch: streamlit run dashboard.py
"""

import os

# Disable tokenizer parallelism before any imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pickle
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml
from yaml.loader import SafeLoader

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Discovery & Config Helpers
# ---------------------------------------------------------------------------


def _list_data_files():
    """Return list of (make, model) tuples that have data/*.pkl files."""
    pairs = []
    data_dir = "data"
    if not os.path.isdir(data_dir):
        return pairs
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith("_data.pkl"):
            stem = fname.replace("_data.pkl", "")
            parts = stem.split("_", 1)
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def _list_model_dirs():
    """Return list of (make, model) tuples that have AutoGluon model dirs."""
    pairs = []
    models_dir = "models"
    if not os.path.isdir(models_dir):
        return pairs
    for name in sorted(os.listdir(models_dir)):
        path = os.path.join(models_dir, name)
        if not os.path.isdir(path):
            continue
        # Skip quantile dirs and special dirs
        if name.endswith("_quantile") or name == "general_model":
            continue
        # Must contain predictor.pkl (AutoGluon signature)
        if os.path.exists(os.path.join(path, "predictor.pkl")):
            parts = name.split("_", 1)
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    return pairs


@st.cache_data
def discover_available_models():
    """Return dict of make -> [models] that have both data and trained models."""
    data_pairs = set(_list_data_files())
    model_pairs = set(_list_model_dirs())
    valid = sorted(data_pairs & model_pairs)
    result = {}
    for make, model in valid:
        result.setdefault(make, []).append(model)
    return result


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=SafeLoader)


# ---------------------------------------------------------------------------
# Cached Loaders
# ---------------------------------------------------------------------------


@st.cache_resource
def load_predictor(path):
    from autogluon.tabular import TabularPredictor

    return TabularPredictor.load(path)


@st.cache_resource
def load_quantile_predictor(path):
    from autogluon.tabular import TabularPredictor

    return TabularPredictor.load(path)


@st.cache_resource
def load_embedding_model(name):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


@st.cache_data
def load_raw_data(path):
    return pd.read_pickle(path)


@st.cache_data
def get_processed_data(make, model_name):
    """Load raw data and run feature engineering (cached)."""
    from utils import load_data, load_config
    from feature_engineering import feature_engineering

    config = load_config()
    # Update config with make/model for classifier lookup
    config["make"] = make
    config["model"] = model_name
    data = load_data(f"data/{make}_{model_name}_data.pkl")
    try:
        data = data[data["BidCount"].isnull()].reset_index(drop=True)
    except KeyError:
        pass
    data = feature_engineering(data, config)
    return data


@st.cache_data
def get_oos_split(make, model_name):
    """Reproduce the 60/40 train/OOS split used during training."""
    data = get_processed_data(make, model_name)
    data = data.drop("ListingId", axis=1, errors="ignore")
    oos = data.sample(frac=0.4, random_state=42)
    train = data.drop(oos.index)
    return train, oos


@st.cache_data
def get_leaderboard(_pred_path, make, model_name):
    """Get AutoGluon leaderboard evaluated on OOS data."""
    predictor = load_predictor(_pred_path)
    _, oos = get_oos_split(make, model_name)
    return predictor.leaderboard(data=oos, silent=True)


@st.cache_data
def get_feature_importance(_pred_path, make, model_name):
    """Get feature importance scores from OOS data."""
    predictor = load_predictor(_pred_path)
    _, oos = get_oos_split(make, model_name)
    return predictor.feature_importance(data=oos, silent=True)


# ---------------------------------------------------------------------------
# Tab 1: Valuation Card
# ---------------------------------------------------------------------------


def _get_available_make_models():
    """Get list of available make/model combinations from trained models."""
    models_dir = "models"
    available = []
    if os.path.isdir(models_dir):
        for name in sorted(os.listdir(models_dir)):
            path = os.path.join(models_dir, name)
            if not os.path.isdir(path):
                continue
            if name.endswith("_quantile") or name == "general_model":
                continue
            if os.path.exists(os.path.join(path, "predictor.pkl")):
                parts = name.split("_", 1)
                if len(parts) == 2:
                    available.append((parts[0], parts[1]))
    return available


def _has_general_model():
    """Check if general model exists."""
    return os.path.exists("models/general_model/predictor.pkl")


def render_paste_listing_section():
    """Render the paste listing extraction and valuation UI."""
    from feature_engineering import build_data_point, classify_colour, classify_stereo
    from inference import predict_with_confidence

    with st.expander("Paste a Facebook Listing", expanded=False):
        listing_text = st.text_area(
            "Paste listing text here",
            height=200,
            placeholder="Paste the full text from a Facebook Marketplace car listing...",
            key="listing_text_input",
        )

        if st.button("Extract & Value", key="extract_button"):
            if not listing_text.strip():
                st.error("Please paste some listing text first.")
                return

            try:
                from llm_extraction import extract_car_params

                with st.spinner("Extracting car details..."):
                    extracted = extract_car_params(listing_text)
                st.session_state["extracted_listing"] = extracted
                st.session_state["extraction_error"] = None
            except ValueError as e:
                st.session_state["extraction_error"] = str(e)
                st.session_state.pop("extracted_listing", None)
            except Exception as e:
                st.session_state["extraction_error"] = f"Extraction failed: {e}"
                st.session_state.pop("extracted_listing", None)

        # Show extraction error if any
        if st.session_state.get("extraction_error"):
            st.error(st.session_state["extraction_error"])

        # Show extracted data with editable fields
        if "extracted_listing" in st.session_state:
            extracted = st.session_state["extracted_listing"]

            # Confidence warning
            if extracted.confidence == "low":
                st.warning(
                    "Low confidence extraction. Please review all fields carefully."
                )
            elif extracted.confidence == "medium":
                st.info("Medium confidence. Review fields before valuing.")

            if extracted.extraction_notes:
                st.caption(f"Notes: {extracted.extraction_notes}")

            st.markdown("#### Extracted Details (editable)")

            # Check model availability
            available_models = _get_available_make_models()
            has_general = _has_general_model()

            # Find if we have a specific model
            specific_model_exists = any(
                m.lower() == extracted.make.lower()
                and mdl.lower() == extracted.model.lower()
                for m, mdl in available_models
            )

            if not specific_model_exists:
                if has_general:
                    st.warning(
                        f"No specific model for {extracted.make} {extracted.model}. "
                        "Using general model (less accurate)."
                    )
                else:
                    st.error(
                        f"No model available for {extracted.make} {extracted.model}. "
                        f"Available models: {', '.join(f'{m} {mdl}' for m, mdl in available_models)}"
                    )

            # Editable form
            col1, col2 = st.columns(2)

            with col1:
                make = st.text_input("Make", value=extracted.make, key="edit_make")
                model = st.text_input("Model", value=extracted.model, key="edit_model")
                year = st.number_input(
                    "Year",
                    value=extracted.year,
                    min_value=1990,
                    max_value=2026,
                    key="edit_year",
                )
                odometer = st.number_input(
                    "Odometer (km)",
                    value=extracted.odometer,
                    min_value=0,
                    key="edit_odometer",
                )
                engine_cc = st.number_input(
                    "Engine Size (cc)",
                    value=extracted.engine_size_cc or 1500,
                    min_value=500,
                    max_value=8000,
                    key="edit_engine",
                )

            with col2:
                fuel_type = st.selectbox(
                    "Fuel Type",
                    ["Petrol", "Diesel", "Electric", "Hybrid"],
                    index=["Petrol", "Diesel", "Electric", "Hybrid"].index(
                        extracted.fuel_type
                    ),
                    key="edit_fuel",
                )
                transmission = st.selectbox(
                    "Transmission",
                    ["Automatic", "Manual"],
                    index=["Automatic", "Manual"].index(extracted.transmission),
                    key="edit_transmission",
                )
                cylinders = st.selectbox(
                    "Cylinders",
                    [3, 4, 6, 8, 10, 12],
                    index=[3, 4, 6, 8, 10, 12].index(extracted.cylinders)
                    if extracted.cylinders in [3, 4, 6, 8, 10, 12]
                    else 1,
                    key="edit_cylinders",
                )
                is_4wd = st.checkbox(
                    "4WD / AWD", value=extracted.is_4wd, key="edit_4wd"
                )
                colour = st.text_input(
                    "Exterior Colour",
                    value=extracted.exterior_colour,
                    key="edit_colour",
                )

            # Region selector
            regions = [
                "Auckland",
                "Bay of Plenty",
                "Canterbury",
                "Gisborne",
                "Hawke's Bay",
                "Manawatu",
                "Marlborough",
                "Nelson Bays",
                "Northland",
                "Otago",
                "Southland",
                "Taranaki",
                "Timaru - Oamaru",
                "Waikato",
                "Wairarapa",
                "Wellington",
                "West Coast",
                "Whanganui",
            ]
            default_region_idx = 0
            if extracted.region and extracted.region in regions:
                default_region_idx = regions.index(extracted.region)
            region = st.selectbox(
                "Region", regions, index=default_region_idx, key="edit_region"
            )

            # Asking price display
            if extracted.asking_price:
                st.metric("Asking Price", f"${extracted.asking_price:,}")

            # Run valuation button
            can_value = specific_model_exists or has_general
            if st.button(
                "Run Valuation", disabled=not can_value, key="run_valuation_button"
            ):
                # Determine which make/model to use for prediction
                if specific_model_exists:
                    pred_make, pred_model = next(
                        (m, mdl)
                        for m, mdl in available_models
                        if m.lower() == make.lower() and mdl.lower() == model.lower()
                    )
                else:
                    pred_make, pred_model = (
                        available_models[0]
                        if available_models
                        else ("General", "Model")
                    )

                model_settings = _load_yaml("model_settings.yml")
                embedding_model = load_embedding_model(
                    model_settings["text_embedding_model"]
                )

                # Classify colour and stereo
                colour_label = classify_colour(
                    pd.DataFrame({"ExteriorColour": [colour]}), embedding_model
                ).iloc[0]["colour_label"]
                stereo_label = classify_stereo(
                    pd.DataFrame({"StereoDescription": ["Standard"]}), embedding_model
                ).iloc[0]["stereo_label"]

                # Build data point
                data_point = build_data_point(
                    region,
                    engine_cc,
                    odometer,
                    year,
                    fuel_type,
                    transmission,
                    cylinders,
                    colour_label,
                    stereo_label,
                    1 if is_4wd else 0,
                )

                try:
                    result = predict_with_confidence(data_point, pred_make, pred_model)
                    st.session_state["paste_valuation_result"] = {
                        "result": result,
                        "make": make,
                        "model": model,
                        "year": year,
                        "asking_price": extracted.asking_price,
                        "used_general": not specific_model_exists,
                    }
                except Exception as e:
                    st.error(f"Valuation failed: {e}")

            # Display valuation result
            if "paste_valuation_result" in st.session_state:
                val = st.session_state["paste_valuation_result"]
                result = val["result"]
                estimate = result["estimate"]

                st.markdown("---")
                st.subheader(f"Valuation: {val['year']} {val['make']} {val['model']}")

                if val.get("used_general"):
                    st.caption("(Valued using general model)")

                col1, col2 = st.columns(2)
                col1.metric("Estimated Value", f"${estimate:,.0f}")

                if val.get("asking_price"):
                    diff = val["asking_price"] - estimate
                    diff_pct = (diff / estimate) * 100
                    col2.metric(
                        "vs Asking Price",
                        f"${diff:+,.0f}",
                        delta=f"{diff_pct:+.1f}%",
                        delta_color="inverse",
                    )

                # Tail adjustment indicator
                if result.get("tail") == "low":
                    st.warning("Tail-adjusted (low end)")
                elif result.get("tail") == "high":
                    st.info("Tail-adjusted (high end)")

                # Confidence intervals
                if "ci_80" in result or "ci_50" in result:
                    import plotly.graph_objects as go

                    fig = go.Figure()
                    if "ci_80" in result:
                        lo80, hi80 = result["ci_80"]
                        fig.add_trace(
                            go.Bar(
                                y=["Estimate"],
                                x=[hi80 - lo80],
                                base=[lo80],
                                orientation="h",
                                name="80% CI",
                                marker_color="rgba(99,110,250,0.25)",
                            )
                        )
                    if "ci_50" in result:
                        lo50, hi50 = result["ci_50"]
                        fig.add_trace(
                            go.Bar(
                                y=["Estimate"],
                                x=[hi50 - lo50],
                                base=[lo50],
                                orientation="h",
                                name="50% CI",
                                marker_color="rgba(99,110,250,0.5)",
                            )
                        )
                    fig.add_trace(
                        go.Scatter(
                            x=[estimate],
                            y=["Estimate"],
                            mode="markers",
                            marker=dict(size=14, color="red", symbol="diamond"),
                            name="Point Estimate",
                        )
                    )
                    fig.update_layout(
                        height=100,
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis_title="Price ($)",
                        barmode="overlay",
                        showlegend=True,
                        legend=dict(orientation="h", y=-0.5),
                    )
                    st.plotly_chart(fig, use_container_width=True)


def render_valuation_page(make, model_name):
    """Render the valuation card tab."""
    from feature_engineering import build_data_point, classify_colour, classify_stereo
    from inference import predict_with_confidence, calculate_depreciation

    # Paste listing section at the top
    render_paste_listing_section()

    st.markdown("---")
    st.markdown("### Value from Config File")

    config_path = "car_to_value.yml"
    if not os.path.exists(config_path):
        st.warning("No `car_to_value.yml` found. Create one to use this section.")
        return

    car_specs = _load_yaml(config_path)

    v_make = car_specs["make"]
    v_model = car_specs["model"]

    model_path = f"models/{v_make}_{v_model}"
    if not os.path.isdir(model_path):
        st.error(f"No model found at `{model_path}`")
        return

    predictor = load_predictor(model_path)

    # Parse engine size
    engine_size_str = str(car_specs["engine_size"]).lower()
    if "l" in engine_size_str:
        engine_size = float(engine_size_str.replace("l", "").strip()) * 1000
    elif "cc" in engine_size_str:
        engine_size = float(engine_size_str.replace("cc", "").strip())
    else:
        engine_size = float(engine_size_str)

    region = car_specs["region"]
    year = int(car_specs["year"])
    odometer = float(car_specs["odometer"])
    fuel_type = car_specs["fuel_type"]
    transmission = car_specs["transmission"]
    cylinder = int(car_specs["cylinders"])
    exterior_colour = car_specs.get("exterior_colour", "Unknown")
    stereo_description = car_specs.get("stereo_description", "Standard")
    is4Wd = 1 if car_specs.get("is_4wd", False) else 0

    # Classify colour/stereo
    model_settings = _load_yaml("model_settings.yml")
    embedding_model = load_embedding_model(model_settings["text_embedding_model"])

    colour_label = classify_colour(
        pd.DataFrame({"ExteriorColour": [exterior_colour]}), embedding_model
    ).iloc[0]["colour_label"]
    stereo_label = classify_stereo(
        pd.DataFrame({"StereoDescription": [stereo_description]}), embedding_model
    ).iloc[0]["stereo_label"]

    data_point = build_data_point(
        region,
        engine_size,
        odometer,
        year,
        fuel_type,
        transmission,
        cylinder,
        colour_label,
        stereo_label,
        is4Wd,
    )

    # Predict
    result = predict_with_confidence(data_point, v_make, v_model)
    estimate = result["estimate"]

    # --- Layout ---
    st.subheader(f"{year} {v_make} {v_model}")
    specs_text = (
        f"{engine_size:.0f}cc | {fuel_type} | {transmission} | "
        f"{cylinder} cyl | {odometer:,.0f} km | {region} | "
        f"{'4WD' if is4Wd else '2WD'} | {exterior_colour}"
    )
    st.caption(specs_text)

    st.metric("Estimated Value", f"${estimate:,.0f}")

    # Tail adjustment indicator
    if result.get("tail") == "low":
        st.warning(
            "Tail-adjusted (low end) — prediction blended with low-tail Ridge model."
        )
    elif result.get("tail") == "high":
        st.info(
            "Tail-adjusted (high end) — prediction blended with high-tail Ridge model."
        )

    # Confidence interval chart
    if "ci_80" in result or "ci_50" in result:
        st.markdown("#### Confidence Intervals")
        fig = go.Figure()

        # 80% CI band
        if "ci_80" in result:
            lo80, hi80 = result["ci_80"]
            fig.add_trace(
                go.Bar(
                    y=["Estimate"],
                    x=[hi80 - lo80],
                    base=[lo80],
                    orientation="h",
                    name="80% CI",
                    marker_color="rgba(99,110,250,0.25)",
                    hovertemplate="80%% CI: $%{base:,.0f} – $%{customdata:,.0f}<extra></extra>",
                    customdata=[hi80],
                )
            )

        # 50% CI band
        if "ci_50" in result:
            lo50, hi50 = result["ci_50"]
            fig.add_trace(
                go.Bar(
                    y=["Estimate"],
                    x=[hi50 - lo50],
                    base=[lo50],
                    orientation="h",
                    name="50% CI",
                    marker_color="rgba(99,110,250,0.5)",
                    hovertemplate="50%% CI: $%{base:,.0f} – $%{customdata:,.0f}<extra></extra>",
                    customdata=[hi50],
                )
            )

        # Point estimate marker
        fig.add_trace(
            go.Scatter(
                x=[estimate],
                y=["Estimate"],
                mode="markers",
                marker=dict(size=14, color="red", symbol="diamond"),
                name="Point Estimate",
                hovertemplate="Estimate: $%{x:,.0f}<extra></extra>",
            )
        )

        fig.update_layout(
            height=120,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="Price ($)",
            barmode="overlay",
            showlegend=True,
            legend=dict(orientation="h", y=-0.4),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Depreciation forecast
    st.markdown("#### 12-Month Depreciation Forecast")
    depreciation = calculate_depreciation(predictor, data_point, estimate)

    if not depreciation["reliable"]:
        st.warning("Depreciation estimates may be unreliable for this vehicle.")

    dep_data = pd.DataFrame(
        {
            "Scenario": ["Current", "+10,000 km / +1 yr", "Parked / +1 yr"],
            "Value": [estimate, depreciation["value_10k"], depreciation["value_0k"]],
        }
    )
    fig_dep = px.bar(
        dep_data,
        x="Scenario",
        y="Value",
        text=dep_data["Value"].apply(lambda v: f"${v:,.0f}"),
        color="Scenario",
        color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96"],
    )
    fig_dep.update_layout(
        showlegend=False,
        yaxis_title="Estimated Value ($)",
        height=350,
        margin=dict(t=10),
    )
    fig_dep.update_traces(textposition="outside")
    st.plotly_chart(fig_dep, use_container_width=True)

    # Regional comparison
    comparison_region = car_specs.get("comparison_region")
    if comparison_region and comparison_region != region:
        st.markdown("#### Regional Comparison")
        comparison_data_point = build_data_point(
            comparison_region,
            engine_size,
            odometer,
            year,
            fuel_type,
            transmission,
            cylinder,
            colour_label,
            stereo_label,
            is4Wd,
        )
        comparison_price = predictor.predict(comparison_data_point).iloc[0]
        delta = comparison_price - estimate
        delta_pct = (delta / estimate) * 100

        col1, col2 = st.columns(2)
        col1.metric(f"Value in {region}", f"${estimate:,.0f}")
        col2.metric(
            f"Value in {comparison_region}",
            f"${comparison_price:,.0f}",
            delta=f"${delta:+,.0f} ({delta_pct:+.1f}%)",
        )


# ---------------------------------------------------------------------------
# Tab 2: Model Diagnostics
# ---------------------------------------------------------------------------


def render_diagnostics_page(make, model_name):
    """Render model diagnostics tab."""
    from utils import load_model_metadata

    model_path = f"models/{make}_{model_name}"
    if not os.path.isdir(model_path):
        st.error(f"No model at `{model_path}`")
        return

    # Metadata header
    meta = load_model_metadata(model_path)
    col1, col2, col3 = st.columns(3)
    col1.metric("Training Date", meta.get("training_date", "unknown")[:10])
    col2.metric("Training Samples", meta.get("n_training_samples", "?"))

    predictor = load_predictor(model_path)
    best_model = predictor.model_best
    col3.metric("Best Model", best_model)

    # Leaderboard
    st.markdown("#### AutoGluon Leaderboard (OOS)")
    lb = get_leaderboard(model_path, make, model_name)
    st.dataframe(lb, use_container_width=True)

    # Feature importance
    st.markdown("#### Feature Importance (OOS)")
    fi = get_feature_importance(model_path, make, model_name)
    fi_sorted = fi.sort_values("importance", ascending=True)
    fig_fi = px.bar(
        fi_sorted,
        x="importance",
        y=fi_sorted.index,
        orientation="h",
        error_x="stddev" if "stddev" in fi_sorted.columns else None,
        labels={"importance": "Importance", "index": "Feature"},
    )
    fig_fi.update_layout(height=max(300, len(fi_sorted) * 30), margin=dict(t=10))
    st.plotly_chart(fig_fi, use_container_width=True)

    # OOS predictions
    _, oos = get_oos_split(make, model_name)
    preds = predictor.predict(oos)
    actuals = oos["price"]
    residuals = actuals.values - preds.values

    # Evaluation metrics
    st.markdown("#### Evaluation Metrics (OOS)")
    eval_results = predictor.evaluate(oos, silent=True)
    metrics_df = pd.DataFrame(
        {"Metric": list(eval_results.keys()), "Value": list(eval_results.values())}
    )
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Predicted vs Actual scatter
    st.markdown("#### Predicted vs Actual")
    scatter_df = pd.DataFrame(
        {
            "Actual": actuals.values,
            "Predicted": preds.values,
            "Residual": np.abs(residuals),
        }
    )
    fig_pva = px.scatter(
        scatter_df,
        x="Actual",
        y="Predicted",
        color="Residual",
        color_continuous_scale="RdYlGn_r",
        labels={"Residual": "|Residual|"},
    )
    # y=x reference line
    xy_min = min(scatter_df["Actual"].min(), scatter_df["Predicted"].min())
    xy_max = max(scatter_df["Actual"].max(), scatter_df["Predicted"].max())
    fig_pva.add_trace(
        go.Scatter(
            x=[xy_min, xy_max],
            y=[xy_min, xy_max],
            mode="lines",
            line=dict(dash="dash", color="grey"),
            showlegend=False,
        )
    )
    fig_pva.update_layout(height=450, margin=dict(t=10))
    st.plotly_chart(fig_pva, use_container_width=True)

    # Residuals plots side by side
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Residuals Distribution")
        fig_hist = px.histogram(
            x=residuals,
            nbins=30,
            labels={"x": "Actual − Predicted ($)"},
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
        fig_hist.update_layout(height=350, margin=dict(t=10), showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        st.markdown("#### Residuals vs Predicted")
        fig_rvp = px.scatter(
            x=preds.values,
            y=residuals,
            labels={"x": "Predicted ($)", "y": "Residual ($)"},
        )
        fig_rvp.add_hline(y=0, line_dash="dash", line_color="red")
        fig_rvp.update_layout(height=350, margin=dict(t=10))
        st.plotly_chart(fig_rvp, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3: Market Overview
# ---------------------------------------------------------------------------


def render_market_page(make, model_name):
    """Render market overview tab."""
    data = get_processed_data(make, model_name)

    if "price" not in data.columns:
        st.error("No price column in processed data.")
        return

    prices = data["price"]

    # Summary metrics
    st.markdown("#### Market Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Listings", f"{len(data):,}")
    c2.metric("Median Price", f"${prices.median():,.0f}")
    c3.metric("Price Range", f"${prices.min():,.0f} – ${prices.max():,.0f}")
    avg_age = data["Age"].mean() if "Age" in data.columns else None
    c4.metric("Average Age", f"{avg_age:.1f} yrs" if avg_age is not None else "N/A")

    # Price distribution
    st.markdown("#### Price Distribution")
    q25, q50, q75 = prices.quantile([0.25, 0.5, 0.75])
    fig_price = px.histogram(data, x="price", nbins=40, labels={"price": "Price ($)"})
    for val, label, color in [
        (q25, "Q25", "orange"),
        (q50, "Median", "red"),
        (q75, "Q75", "orange"),
    ]:
        fig_price.add_vline(
            x=val,
            line_dash="dash",
            line_color=color,
            annotation_text=f"{label}: ${val:,.0f}",
        )
    fig_price.update_layout(height=350, margin=dict(t=30), showlegend=False)
    st.plotly_chart(fig_price, use_container_width=True)

    # Price by Year
    if "Year" in data.columns:
        st.markdown("#### Price by Year")
        chart_type = st.radio(
            "Chart type", ["Box", "Scatter"], horizontal=True, key="yr_type"
        )
        if chart_type == "Box":
            fig_yr = px.box(data, x="Year", y="price", labels={"price": "Price ($)"})
        else:
            fig_yr = px.scatter(
                data, x="Year", y="price", opacity=0.5, labels={"price": "Price ($)"}
            )
        fig_yr.update_layout(height=400, margin=dict(t=10))
        st.plotly_chart(fig_yr, use_container_width=True)

    # Price by Odometer
    if "Odometer" in data.columns:
        st.markdown("#### Price by Odometer")
        fig_odo = px.scatter(
            data,
            x="Odometer",
            y="price",
            opacity=0.4,
            trendline="lowess",
            labels={"price": "Price ($)", "Odometer": "Odometer (km)"},
        )
        fig_odo.update_layout(height=400, margin=dict(t=10))
        st.plotly_chart(fig_odo, use_container_width=True)

    # Price by Region
    if "Region" in data.columns:
        st.markdown("#### Price by Region")
        region_order = (
            data.groupby("Region")["price"]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )
        fig_reg = px.box(
            data,
            x="Region",
            y="price",
            category_orders={"Region": region_order},
            labels={"price": "Price ($)"},
        )
        fig_reg.update_layout(height=400, margin=dict(t=10))
        st.plotly_chart(fig_reg, use_container_width=True)

    # Category distributions
    st.markdown("#### Category Distributions")
    cat_cols = [
        c
        for c in ["Region", "Fuel", "Transmission", "colour_label", "stereo_label"]
        if c in data.columns
    ]
    if cat_cols:
        cols = st.columns(min(len(cat_cols), 3))
        for i, col_name in enumerate(cat_cols):
            with cols[i % len(cols)]:
                counts = data[col_name].value_counts()
                fig_bar = px.bar(
                    x=counts.index,
                    y=counts.values,
                    labels={"x": col_name, "y": "Count"},
                )
                fig_bar.update_layout(height=300, margin=dict(t=10), showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

    # Tail info
    model_path = f"models/{make}_{model_name}"
    tail_path = os.path.join(model_path, "tail_info.pkl")
    if os.path.exists(tail_path):
        st.markdown("#### Tail Model Info")
        with open(tail_path, "rb") as f:
            tail_info = pickle.load(f)
        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("Q25 Boundary", f"${tail_info['q25']:,.0f}")
        tc2.metric("Q75 Boundary", f"${tail_info['q75']:,.0f}")
        low_status = (
            f"Yes (n={tail_info.get('low_n', '?')})"
            if tail_info.get("low_model")
            else f"No (n={tail_info.get('low_n', '?')})"
        )
        high_status = (
            f"Yes (n={tail_info.get('high_n', '?')})"
            if tail_info.get("high_model")
            else f"No (n={tail_info.get('high_n', '?')})"
        )
        tc1.caption(f"Low tail model: {low_status}")
        tc3.caption(f"High tail model: {high_status}")


# ---------------------------------------------------------------------------
# Tab 4: Undervalued Finder
# ---------------------------------------------------------------------------


def render_undervalued_page():
    """Render undervalued car finder tab."""
    available = discover_available_models()
    if not available:
        st.warning("No trained models found.")
        return

    with st.form("undervalued_form"):
        st.markdown("#### Search Configuration")
        col1, col2 = st.columns(2)

        with col1:
            all_cars = []
            for mk, models in available.items():
                for mdl in models:
                    all_cars.append(f"{mk} {mdl}")
            selected_cars = st.multiselect(
                "Cars to search",
                all_cars,
                default=all_cars,
            )

        with col2:
            min_disc = st.slider("Min discount %", 0, 100, 40)
            max_disc = st.slider("Max discount %", 0, 100, 80)

        comparison_region = st.text_input(
            "Comparison region (optional)",
            placeholder="e.g. Auckland",
        )

        submitted = st.form_submit_button("Search", type="primary")

    if submitted and selected_cars:
        from find_undervalued import find_undervalued

        all_results = []
        progress = st.progress(0, text="Searching...")

        for i, car_str in enumerate(selected_cars):
            parts = car_str.split(" ", 1)
            mk, mdl = parts[0], parts[1]
            progress.progress(
                (i + 1) / len(selected_cars),
                text=f"Searching {mk} {mdl}...",
            )
            result = find_undervalued(
                mk,
                mdl,
                min_disc,
                max_disc,
                filters=None,
                comparison_region=comparison_region or None,
            )
            if result is not None and len(result) > 0:
                all_results.append(result)

        progress.empty()

        if not all_results:
            st.info("No undervalued cars found matching criteria.")
            st.session_state.pop("undervalued_results", None)
            st.session_state.pop("undervalued_comparison_region", None)
            return

        combined = pd.concat(all_results, ignore_index=True)
        combined = combined.sort_values("discount_pct", ascending=False)
        st.session_state["undervalued_results"] = combined
        st.session_state["undervalued_comparison_region"] = comparison_region

    # Display results from session state
    if "undervalued_results" not in st.session_state:
        return

    combined = st.session_state["undervalued_results"]
    comparison_region = st.session_state.get("undervalued_comparison_region", "")

    # --- Post-search filters ---
    st.markdown("#### Filter Results")
    fc1, fc2, fc3 = st.columns(3)

    odo_min = int(combined["Odometer"].min()) if "Odometer" in combined.columns else 0
    odo_max = (
        int(combined["Odometer"].max()) if "Odometer" in combined.columns else 300000
    )
    age_min = int(combined["Age"].min()) if "Age" in combined.columns else 0
    age_max = int(combined["Age"].max()) if "Age" in combined.columns else 30
    price_min = int(combined["prediction_label"].min())
    price_max = int(combined["prediction_label"].max())

    with fc1:
        odo_range = st.slider(
            "Odometer (km)",
            odo_min,
            odo_max,
            (odo_min, odo_max),
            step=10000,
            key="uv_odo_filter",
        )
    with fc2:
        age_range = st.slider(
            "Age (years)",
            age_min,
            age_max,
            (age_min, age_max),
            step=1,
            key="uv_age_filter",
        )
    with fc3:
        price_range = st.slider(
            "Est. Price ($)",
            price_min,
            price_max,
            (price_min, price_max),
            step=1000,
            key="uv_price_filter",
        )

    filtered = combined.copy()
    if "Odometer" in filtered.columns:
        filtered = filtered[
            (filtered["Odometer"] >= odo_range[0])
            & (filtered["Odometer"] <= odo_range[1])
        ]
    if "Age" in filtered.columns:
        filtered = filtered[
            (filtered["Age"] >= age_range[0]) & (filtered["Age"] <= age_range[1])
        ]
    filtered = filtered[
        (filtered["prediction_label"] >= price_range[0])
        & (filtered["prediction_label"] <= price_range[1])
    ]

    # Summary stats
    st.markdown("#### Results")
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Cars Found", len(filtered))
    if len(filtered) > 0:
        sc2.metric("Avg Discount", f"{filtered['discount_pct'].mean():.1f}%")
        sc3.metric(
            "Price Range",
            f"${filtered['price'].min():,.0f} – ${filtered['price'].max():,.0f}",
        )
    else:
        sc2.metric("Avg Discount", "—")
        sc3.metric("Price Range", "—")

    if len(filtered) == 0:
        st.info("No cars match the current filters.")
        return

    # Build display dataframe
    display = filtered.copy()

    # TradeMe link
    if "ListingId" in display.columns:
        display["URL"] = display["ListingId"].apply(
            lambda lid: f"https://www.trademe.co.nz/a/{lid}"
        )

    display_cols = ["Make", "Model"]
    if "Title" in display.columns:
        display_cols.append("Title")
    display_cols += ["price", "prediction_label", "discount_pct"]
    if "Region" in display.columns:
        display_cols.append("Region")
    elif "OrigRegion" in display.columns:
        display_cols.append("OrigRegion")
    if comparison_region and "comparison_value" in display.columns:
        display_cols += ["comparison_value", "regional_diff_pct"]
    if "URL" in display.columns:
        display_cols.append("URL")

    display_cols = [c for c in display_cols if c in display.columns]
    display = display[display_cols]

    column_config = {
        "price": st.column_config.NumberColumn("Price", format="$%d"),
        "prediction_label": st.column_config.NumberColumn("Est. Value", format="$%d"),
        "discount_pct": st.column_config.NumberColumn("Discount %", format="%.1f%%"),
    }
    if "URL" in display.columns:
        column_config["URL"] = st.column_config.LinkColumn(
            "Listing", display_text="View"
        )
    if "comparison_value" in display.columns:
        column_config["comparison_value"] = st.column_config.NumberColumn(
            f"{comparison_region} Value", format="$%d"
        )
    if "regional_diff_pct" in display.columns:
        column_config["regional_diff_pct"] = st.column_config.NumberColumn(
            "Region Diff %", format="%+.1f%%"
        )
    if "OrigRegion" in display.columns:
        column_config["OrigRegion"] = "Region"

    st.dataframe(
        display,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(layout="wide", page_title="Car Valuation Dashboard")
    st.title("Car Valuation Dashboard")

    # Sidebar — make/model selector
    available = discover_available_models()

    if not available:
        st.error(
            "No trained models found. Train a model first with "
            "`python valuation_model.py` or `python find_undervalued.py --train`."
        )
        return

    with st.sidebar:
        st.header("Model Selection")
        makes = list(available.keys())
        selected_make = st.selectbox("Make", makes)
        models = available[selected_make]
        selected_model = st.selectbox("Model", models)

        # Show metadata
        model_path = f"models/{selected_make}_{selected_model}"
        meta_path = os.path.join(model_path, "metadata.yml")
        if os.path.exists(meta_path):
            meta = _load_yaml(meta_path)
            st.caption(f"Trained: {meta.get('training_date', '?')[:10]}")
            st.caption(f"Samples: {meta.get('n_training_samples', '?')}")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Valuation",
            "Model Diagnostics",
            "Market Overview",
            "Undervalued Finder",
        ]
    )

    with tab1:
        render_valuation_page(selected_make, selected_model)

    with tab2:
        render_diagnostics_page(selected_make, selected_model)

    with tab3:
        render_market_page(selected_make, selected_model)

    with tab4:
        render_undervalued_page()


if __name__ == "__main__":
    main()
