"""
Pattern Matching Analysis - Historical Pattern Recognition
A tool to find similar historical patterns in financial markets using DTW and correlation
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.stats import pearsonr
import ta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from translations import TRANSLATIONS


def main():
    """
    Main function that runs the Streamlit application
    """
    st.set_page_config(
        page_title="Pattern Matching Analysis",
        layout="wide",
        page_icon="📊",
        menu_items={
            'Get Help': 'https://github.com/TeknoTrader',
            'Report a bug': 'https://github.com/TeknoTrader/Historycal-Pattern-Matching',
            'About': '# Pattern Matching Analysis\nCreated by Nicola Chimenti\nCS50P Final Project 2026'
        }
    )

    # Lingua predefinita: INGLESE
    if 'lang' not in st.session_state:
        st.session_state['lang'] = 'en'

    lang = render_sidebar()
    st.title(get_text('title', lang))
    st.markdown(get_text('subtitle', lang))

    if st.session_state.get('run_analysis', False):
        execute_analysis(lang)
    elif 'performance_df' in st.session_state:
        render_cached_results(lang)
    else:
        st.info(get_text('configure_sidebar', lang))

    render_footer(lang)


# Calculate pattern similarity using DTW and Pearson correlation
# Returns tuple of (dtw_similarity, dtw_distance, correlation)
def calculate_pattern_similarity(current_normalized, historical_normalized, current_returns, historical_returns):
    try:
        distance, _ = fastdtw(current_normalized.reshape(-1, 1),
                              historical_normalized.reshape(-1, 1),
                              dist=euclidean)

        max_distance = 100
        dtw_similarity = np.exp(-distance / max_distance)

        if len(current_returns) > 1 and len(historical_returns) > 1:
            correlation, _ = pearsonr(current_returns, historical_returns)
        else:
            correlation = 0.0

        return dtw_similarity, distance, correlation
    except:
        return 0.0, float('inf'), 0.0


# Find similar historical patterns using DTW and correlation filters
# Searches from the beginning of the dataset up to pattern_end_date
def find_similar_patterns(df, pattern_length, pattern_end_date, min_dtw_similarity=0.5, min_correlation=0.3):
    similar_periods = []
    close_series = df['Close'].squeeze()

    pattern_end_idx = df.index.get_indexer([pattern_end_date], method='nearest')[0]
    current_start_idx = pattern_end_idx - pattern_length + 1

    if current_start_idx < 0:
        return pd.DataFrame()

    current_prices = close_series.iloc[current_start_idx:pattern_end_idx + 1].values
    current_normalized = (current_prices - current_prices[0]) / current_prices[0] * 100
    current_returns = np.diff(current_prices) / current_prices[:-1] * 100

    for i in range(len(close_series) - pattern_length - 1):
        if i >= current_start_idx - pattern_length:
            continue

        historical_prices = close_series.iloc[i:i + pattern_length].values

        if len(historical_prices) == pattern_length:
            historical_normalized = (historical_prices - historical_prices[0]) / historical_prices[0] * 100
            historical_returns = np.diff(historical_prices) / historical_prices[:-1] * 100

            dtw_similarity, dtw_distance, correlation = calculate_pattern_similarity(
                current_normalized, historical_normalized,
                current_returns, historical_returns
            )

            if dtw_similarity >= min_dtw_similarity and correlation >= min_correlation:
                similar_periods.append({
                    'start_idx': i,
                    'end_idx': i + pattern_length,
                    'dtw_similarity': dtw_similarity,
                    'dtw_distance': dtw_distance,
                    'correlation': correlation,
                    'combined_score': (dtw_similarity + correlation) / 2,
                    'start_date': close_series.index[i],
                    'end_date': close_series.index[i + pattern_length - 1]
                })

    results_df = pd.DataFrame(similar_periods)

    if len(results_df) > 0:
        results_df = results_df.sort_values('combined_score', ascending=False)

    return results_df


# Analyze what happened after similar patterns (simulating LONG positions)
# Calculates returns and excursions for the specified future period length
def analyze_future_performance(df, similar_periods, future_periods):
    results = []
    close_series = df['Close'].squeeze()

    for _, period in similar_periods.iterrows():
        end_idx = period['end_idx']

        if end_idx + future_periods >= len(df):
            continue

        start_price = close_series.iloc[end_idx]
        future_prices = close_series.iloc[end_idx:end_idx + future_periods + 1]

        max_price = future_prices.max()
        max_positive_excursion = ((max_price - start_price) / start_price) * 100

        min_price = future_prices.min()
        max_negative_excursion = ((min_price - start_price) / start_price) * 100

        end_price = future_prices.iloc[-1]
        final_return = ((end_price - start_price) / start_price) * 100

        direction = 'LONG' if final_return > 0 else 'SHORT'

        results.append({
            'start_date': period['start_date'],
            'dtw_similarity': period['dtw_similarity'],
            'dtw_distance': period['dtw_distance'],
            'correlation': period['correlation'],
            'combined_score': period['combined_score'],
            'direction': direction,
            'final_return_%': final_return,
            'max_positive_excursion_%': max_positive_excursion,
            'max_negative_excursion_%': max_negative_excursion,
            'start_idx': period['start_idx'],
            'end_idx': end_idx
        })

    return pd.DataFrame(results)


# Calculate aggregate statistics from performance data
# Returns dictionary with all statistical metrics
def calculate_statistics(performance_df):
    if len(performance_df) == 0:
        return None

    total_cases = len(performance_df)
    long_cases = len(performance_df[performance_df['direction'] == 'LONG'])
    short_cases = len(performance_df[performance_df['direction'] == 'SHORT'])

    stats = {
        'total_matches': total_cases,
        'long_count': long_cases,
        'short_count': short_cases,
        'long_percentage': (long_cases / total_cases * 100) if total_cases > 0 else 0,
        'short_percentage': (short_cases / total_cases * 100) if total_cases > 0 else 0,
        'avg_dtw_similarity': performance_df['dtw_similarity'].mean(),
        'avg_correlation': performance_df['correlation'].mean(),
        'avg_combined_score': performance_df['combined_score'].mean(),
        'avg_return': performance_df['final_return_%'].mean(),
        'median_return': performance_df['final_return_%'].median(),
        'avg_max_positive_excursion': performance_df['max_positive_excursion_%'].mean(),
        'avg_max_negative_excursion': performance_df['max_negative_excursion_%'].mean(),
        'best_case': performance_df['final_return_%'].max(),
        'worst_case': performance_df['final_return_%'].min(),
    }

    if long_cases > 0:
        long_df = performance_df[performance_df['direction'] == 'LONG']
        stats['long_avg_return'] = long_df['final_return_%'].mean()
        stats['long_avg_max_positive_excursion'] = long_df['max_positive_excursion_%'].mean()
        stats['long_avg_max_negative_excursion'] = long_df['max_negative_excursion_%'].mean()

    if short_cases > 0:
        short_df = performance_df[performance_df['direction'] == 'SHORT']
        stats['short_avg_return'] = short_df['final_return_%'].mean()
        stats['short_avg_max_positive_excursion'] = short_df['max_positive_excursion_%'].mean()
        stats['short_avg_max_negative_excursion'] = short_df['max_negative_excursion_%'].mean()

    return stats


# Validate ticker symbol by attempting to fetch data
# Returns tuple (is_valid, error_message)
def validate_ticker(symbol):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")

        if hist.empty:
            return False, f"No data available for ticker '{symbol}'"

        return True, None
    except Exception as e:
        return False, f"Invalid ticker '{symbol}': {str(e)}"


# Get translated text for given key and language
def get_text(key, lang='en'):
    return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)


# Render footer with author information and links
def render_footer(lang):
    st.markdown("---")

    footer_col1, footer_col2, footer_col3 = st.columns(3)

    with footer_col1:
        st.markdown("**👨‍💻 " + get_text('author', lang) + "**")
        st.markdown("Nicola Chimenti")

    with footer_col2:
        st.markdown("**🔗 " + get_text('links', lang) + "**")
        st.markdown(f"🌐 [{get_text('website', lang)}](https://www.nicolachimenti.com)")
        st.markdown(f"💻 [GitHub](https://github.com/TeknoTrader)")

    with footer_col3:
        st.markdown("**📦 " + get_text('project', lang) + "**")
        st.markdown(f"⭐ [{get_text('source_code', lang)}](https://github.com/TeknoTrader/Historycal-Pattern-Matching)")
        st.markdown(f"📚 [CS50P](https://certificates.cs50.io/1fbe53e1-1594-47fb-b311-aa7b3a91a6d6.pdf?size=letter)")

    st.markdown(
        f"<div style='text-align: center; color: gray; font-size: 0.8em; margin-top: 20px;'>"
        f"{get_text('made_with', lang)} ❤️ {get_text('for_cs50p', lang)}<br>"
        f"© 2026 Nicola Chimenti - {get_text('all_rights', lang)}"
        f"</div>",
        unsafe_allow_html=True
    )


# Render sidebar configuration and return selected language
def render_sidebar():
    with st.sidebar:
        lang = st.selectbox(
            get_text('language', st.session_state['lang']),
            options=['it', 'en', 'uk'],
            format_func=lambda x: {'it': '🇮🇹 Italiano', 'en': '🇬🇧 English', 'uk': '🇺🇦 Українська'}[x],
            index=['it', 'en', 'uk'].index(st.session_state['lang'])
        )

        if lang != st.session_state['lang']:
            st.session_state['lang'] = lang
            st.rerun()

        st.header(get_text('config', lang))

        st.session_state['symbol'] = st.text_input(get_text('symbol', lang), value="AAPL")
        st.session_state['timeframe'] = st.selectbox(get_text('timeframe', lang),
                                                     options=['1d', '1h', '4h', '1wk', '1mo', '5m', '15m', '30m'],
                                                     index=0)

        st.subheader(get_text('historical_data', lang))
        use_custom_start = st.checkbox(get_text('custom_start', lang))

        if use_custom_start:
            st.session_state['start_date'] = st.date_input(
                get_text('start_date', lang),
                value=datetime.now().date() - timedelta(days=365 * 2),
                max_value=datetime.now().date()
            )
        else:
            st.session_state['start_date'] = None

        st.subheader(get_text('pattern_to_analyze', lang))

        use_custom_end = st.checkbox(get_text('custom_pattern_end', lang))
        if use_custom_end:
            st.session_state['pattern_end_date'] = st.date_input(
                get_text('pattern_end_date', lang),
                value=datetime.now().date(),
                max_value=datetime.now().date()
            )
        else:
            st.session_state['pattern_end_date'] = datetime.now().date()

        period_options = [get_text('days', lang), get_text('weeks', lang), get_text('candles', lang)]
        st.session_state['period_type'] = st.selectbox(get_text('period_type', lang), options=period_options)
        st.session_state['period_length'] = st.number_input(
            f"{get_text('number_of', lang)} {st.session_state['period_type']}",
            min_value=3, max_value=1000, value=10
        )

        st.subheader(get_text('future_horizon', lang))
        st.session_state['future_type'] = st.selectbox(
            get_text('period_type', lang) + " (Future)", options=period_options, key='future'
        )
        st.session_state['future_length'] = st.number_input(
            f"{get_text('number_of', lang)} {st.session_state['future_type']}",
            min_value=1, max_value=1000, value=7, key='future_len'
        )

        st.subheader(get_text('similarity_criteria', lang))
        st.session_state['min_dtw'] = st.slider(get_text('min_dtw_threshold', lang),
                                                min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        st.session_state['min_corr'] = st.slider(get_text('min_corr_threshold', lang),
                                                 min_value=-1.0, max_value=1.0, value=0.3, step=0.05)

        if st.button(get_text('run_analysis', lang), type="primary", use_container_width=True):
            st.session_state['run_analysis'] = True
            st.rerun()

        st.markdown("---")

        with st.expander(f"ℹ️ {get_text('about', lang)}"):
            st.markdown(f"**{get_text('author', lang)}:** Nicola Chimenti")
            st.markdown(f"**{get_text('project', lang)}:** CS50P Final Project")
            st.markdown(f"**{get_text('year', lang)}:** 2026")
            st.markdown("")
            st.markdown(f"🌐 [{get_text('website', lang)}](https://www.nicolachimenti.com)")
            st.markdown(f"💻 [GitHub Profile](https://github.com/TeknoTrader)")
            st.markdown(f"⭐ [{get_text('source_code', lang)}](https://github.com/TeknoTrader/Historycal-Pattern-Matching)")
            st.markdown(f"📧 [Email](mailto:assistenza@nicolachimenti.com)")

    return lang


# Execute the pattern matching analysis with current configuration
def execute_analysis(lang):
    symbol = st.session_state.get('symbol', 'AAPL')

    is_valid, error_msg = validate_ticker(symbol)
    if not is_valid:
        st.error(f"❌ {error_msg}")
        st.session_state['run_analysis'] = False
        return

    with st.spinner(f"{get_text('downloading', lang)} {symbol}..."):
        try:
            start_date = st.session_state.get('start_date')

            if start_date:
                df = yf.download(symbol, start=start_date, end=datetime.now(),
                                 interval=st.session_state['timeframe'], progress=False)
            else:
                df = yf.download(symbol, period="max",
                                 interval=st.session_state['timeframe'], progress=False)

            if df.empty:
                st.error(get_text('no_data', lang))
                st.session_state['run_analysis'] = False
                return

            st.success(f"✅ {len(df)} {get_text('periods', lang)}")

            pattern_length_candles = calculate_candle_length(
                st.session_state['period_type'],
                st.session_state['period_length'],
                st.session_state['timeframe'],
                lang
            )

            future_periods = calculate_candle_length(
                st.session_state['future_type'],
                st.session_state['future_length'],
                st.session_state['timeframe'],
                lang
            )

            pattern_end_date = pd.Timestamp(st.session_state['pattern_end_date'])
            if pattern_end_date not in df.index:
                closest_date = df.index[df.index.get_indexer([pattern_end_date], method='nearest')[0]]
                st.info(f"ℹ️ {get_text('using_closest_date', lang)}: {closest_date.strftime('%Y-%m-%d')}")
                pattern_end_date = closest_date

            with st.spinner(get_text('searching_patterns', lang)):
                similar_periods = find_similar_patterns(
                    df, pattern_length_candles, pattern_end_date,
                    st.session_state['min_dtw'],
                    st.session_state['min_corr']
                )

            if len(similar_periods) == 0:
                st.warning(get_text('no_pattern_found', lang))
                st.session_state['run_analysis'] = False
                return

            st.success(f"✅ {len(similar_periods)} {get_text('patterns_matching', lang)}")

            performance_df = analyze_future_performance(df, similar_periods, future_periods)

            if len(performance_df) > 0:
                st.session_state['performance_df'] = performance_df
                st.session_state['df'] = df
                st.session_state['current_start'] = pattern_end_date
                st.session_state['pattern_length_candles'] = pattern_length_candles

                render_results(performance_df, df, pattern_end_date,
                               pattern_length_candles, symbol, lang)

        except Exception as e:
            st.error(f"Error: {str(e)}")

    st.session_state['run_analysis'] = False


# Render previously calculated results from session state
def render_cached_results(lang):
    st.info(get_text('results_available', lang))
    render_results(
        st.session_state['performance_df'],
        st.session_state['df'],
        st.session_state['current_start'],
        st.session_state['pattern_length_candles'],
        st.session_state.get('symbol', 'AAPL'),
        lang
    )


# Render complete analysis results with charts and statistics
def render_results(performance_df, df, pattern_end_date, pattern_length_candles, symbol, lang):
    stats = calculate_statistics(performance_df)

    st.header(get_text('results_title', lang))

    col1, col2, col3 = st.columns([1, 1.2, 1.2])
    with col1:
        st.metric(get_text('total_matches', lang), stats['total_matches'])
    with col2:
        st.metric(get_text('avg_return', lang), f"{stats['avg_return']:.2f}%")
    with col3:
        st.metric(get_text('median_return', lang), f"{stats['median_return']:.2f}%")

    col1, col2, col3 = st.columns([1.5, 1, 1])
    with col1:
        best_str = f"+{stats['best_case']:.1f}%" if stats['best_case'] >= 0 else f"{stats['best_case']:.1f}%"
        worst_str = f"{stats['worst_case']:.1f}%"
        st.metric(get_text('best_worst', lang), f"{best_str} / {worst_str}")
    with col2:
        st.metric(get_text('avg_pos_excursion', lang), f"+{stats['avg_max_positive_excursion']:.2f}%")
    with col3:
        st.metric(get_text('avg_neg_excursion', lang), f"{stats['avg_max_negative_excursion']:.2f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(get_text('long_statistics', lang))
        st.markdown(f"**{stats['long_count']} {get_text('cases', lang)} ({stats['long_percentage']:.1f}%)**")
        if stats['long_count'] > 0:
            st.metric(get_text('avg_final_return', lang), f"+{stats['long_avg_return']:.2f}%")
            st.metric(get_text('max_pos_excursion', lang), f"+{stats['long_avg_max_positive_excursion']:.2f}%")
            st.metric(get_text('max_neg_excursion', lang), f"{stats['long_avg_max_negative_excursion']:.2f}%")

    with col2:
        st.subheader(get_text('short_statistics', lang))
        st.markdown(f"**{stats['short_count']} {get_text('cases', lang)} ({stats['short_percentage']:.1f}%)**")
        if stats['short_count'] > 0:
            st.metric(get_text('avg_final_return', lang), f"{stats['short_avg_return']:.2f}%")
            st.metric(get_text('max_pos_excursion', lang), f"+{stats['short_avg_max_positive_excursion']:.2f}%")
            st.metric(get_text('max_neg_excursion', lang), f"{stats['short_avg_max_negative_excursion']:.2f}%")

    st.markdown("---")

    st.subheader(get_text('pattern_visualization', lang))
    fig = create_comparison_chart(df, pattern_end_date, pattern_length_candles, performance_df, lang)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(get_text('match_details', lang))

    display_df = performance_df[[
        'start_date', 'direction', 'dtw_similarity', 'correlation', 'combined_score',
        'final_return_%', 'max_positive_excursion_%', 'max_negative_excursion_%'
    ]].copy()

    display_df['start_date'] = pd.to_datetime(display_df['start_date']).dt.strftime('%Y-%m-%d')
    display_df = display_df.round(3)

    display_df = display_df.rename(columns={
        'start_date': get_text('start_date_col', lang),
        'direction': get_text('direction_col', lang),
        'dtw_similarity': get_text('dtw_col', lang),
        'correlation': get_text('corr_col', lang),
        'combined_score': get_text('score_col', lang),
        'final_return_%': get_text('final_return_col', lang),
        'max_positive_excursion_%': get_text('max_esc_pos_col', lang),
        'max_negative_excursion_%': get_text('max_esc_neg_col', lang)
    })

    st.dataframe(display_df, use_container_width=True, height=400)

    export_df = prepare_export(performance_df, lang)
    csv = export_df.to_csv(index=False, sep=';', decimal=',', encoding='utf-8-sig')
    st.download_button(
        label=get_text('download_csv', lang),
        data=csv,
        file_name=f"pattern_analysis_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


# Convert period type and length to number of candles based on timeframe
def calculate_candle_length(period_type, period_length, timeframe, lang):
    period_map = {
        get_text('days', lang): 'days',
        get_text('weeks', lang): 'weeks',
        get_text('candles', lang): 'candles'
    }
    period_type_en = period_map.get(period_type, 'candles')

    if period_type_en == 'days':
        if timeframe == '1d':
            return period_length
        elif timeframe == '1h':
            return period_length * 24
        elif timeframe == '4h':
            return period_length * 6
    elif period_type_en == 'weeks':
        if timeframe == '1d':
            return period_length * 5
        elif timeframe == '1wk':
            return period_length

    return period_length


# Create plotly chart comparing current pattern with historical matches
def create_comparison_chart(df, pattern_end_date, pattern_length, performance_df, lang):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False,
        subplot_titles=(get_text('current_pattern', lang), get_text('similar_patterns', lang)),
        vertical_spacing=0.15, row_heights=[0.4, 0.6]
    )

    close_series = df['Close'].squeeze()
    pattern_end_idx = df.index.get_indexer([pattern_end_date], method='nearest')[0]
    current_start_idx = pattern_end_idx - pattern_length + 1

    current_data = close_series.iloc[current_start_idx:pattern_end_idx + 1]
    current_normalized = (current_data / current_data.iloc[0] - 1) * 100

    fig.add_trace(
        go.Scatter(x=list(range(len(current_normalized))), y=current_normalized,
                   name='Current', line=dict(color='blue', width=4)),
        row=1, col=1
    )

    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta', 'yellow']
    for idx, (_, period) in enumerate(performance_df.head(10).iterrows()):
        historical_data = close_series.iloc[period['start_idx']:period['end_idx']]
        historical_normalized = (historical_data / historical_data.iloc[0] - 1) * 100

        label = f"{period['start_date'].strftime('%Y-%m-%d')} | DTW:{period['dtw_similarity']:.2f}"

        fig.add_trace(
            go.Scatter(x=list(range(len(historical_normalized))), y=historical_normalized,
                       name=label, line=dict(color=colors[idx % len(colors)], width=2), opacity=0.6),
            row=2, col=1
        )

    fig.update_xaxes(title_text="Periods", row=2, col=1)
    fig.update_yaxes(title_text="Change %", row=1, col=1)
    fig.update_yaxes(title_text="Change %", row=2, col=1)
    fig.update_layout(height=900, showlegend=True, hovermode='x unified')

    return fig


# Prepare performance DataFrame for CSV export with proper formatting
# Column names are always in English for better Excel compatibility
def prepare_export(performance_df, lang):
    export_df = performance_df[[
        'start_date', 'direction', 'dtw_similarity', 'correlation',
        'combined_score', 'final_return_%', 'max_positive_excursion_%',
        'max_negative_excursion_%'
    ]].copy()

    export_df['start_date'] = pd.to_datetime(export_df['start_date']).dt.strftime('%Y-%m-%d')

    export_df = export_df.rename(columns={
        'start_date': 'Start Date',
        'direction': 'Direction',
        'dtw_similarity': 'DTW Similarity',
        'correlation': 'Correlation',
        'combined_score': 'Combined Score',
        'final_return_%': 'Final Return %',
        'max_positive_excursion_%': 'Max Positive Excursion %',
        'max_negative_excursion_%': 'Max Negative Excursion %'
    })

    numeric_cols = ['DTW Similarity', 'Correlation', 'Combined Score',
                    'Final Return %', 'Max Positive Excursion %', 'Max Negative Excursion %']
    for col in numeric_cols:
        export_df[col] = export_df[col].round(3)

    return export_df


if __name__ == "__main__":
    main()
