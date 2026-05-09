import streamlit as st
import pandas as pd
import numpy as np
import joblib
from surprise import SVD

st.set_page_config(
    page_title='Hybrid Movie Recommender',
    page_icon='🎬',
    layout='wide'
)


@st.cache_resource
def load_artifacts():
    movies     = pd.read_csv('data/movies_clean.csv')
    ratings    = pd.read_csv('data/ratings_clean.csv')
    cosine_sim = joblib.load('models/cosine_similarity.pkl')
    indices    = joblib.load('models/movie_indices.pkl')
    svd        = joblib.load('models/svd_model.pkl')
    return movies, ratings, cosine_sim, indices, svd


movies, ratings, cosine_sim, indices, svd = load_artifacts()


def hybrid_recommend(user_id, movie_title, top_n=10, alpha=0.5, candidate_pool=50):
    if movie_title not in indices:
        return None

    idx = indices[movie_title]

    sim_scores = sorted(
        enumerate(cosine_sim[idx]),
        key=lambda x: x[1],
        reverse=True
    )[1 : candidate_pool + 1]

    rows = []
    for i, content_score in sim_scores:
        movie_id = int(movies.iloc[i]['movieId'])
        raw_collab   = svd.predict(uid=user_id, iid=movie_id).est
        collab_score = raw_collab / 5.0

        hybrid_score = alpha * content_score + (1 - alpha) * collab_score

        rows.append({
            'Title'        : movies.iloc[i]['title'],
            'Genres'       : movies.iloc[i]['genres'],
            'Content Score': round(content_score, 3),
            'Collab Score' : round(collab_score,  3),
            'Hybrid Score' : round(hybrid_score,  3)
        })

    result = pd.DataFrame(rows)
    result = result.sort_values('Hybrid Score', ascending=False).head(top_n)
    result.reset_index(drop=True, inplace=True)
    result.index += 1
    return result


st.title('🎬 Hybrid Movie Recommendation System')
st.markdown(
    'Combines **Content-Based Filtering** (TF-IDF + Cosine Similarity) '
    'and **Collaborative Filtering** (Surprise SVD — Matrix Factorization) '
    'to generate personalised movie recommendations.'
)
st.divider()

with st.sidebar:
    st.header('Dataset Info')
    st.metric('Movies',  f'{len(movies):,}')
    st.metric('Users',   f'{ratings["userId"].nunique():,}')
    st.metric('Ratings', f'{len(ratings):,}')

    st.divider()
    st.header('Model Info')
    st.markdown(
        '**Content-Based**\n'
        '- TF-IDF on movie genres\n'
        '- Cosine Similarity matrix\n\n'
        '**Collaborative (Surprise SVD)**\n'
        '- scikit-surprise library\n'
        '- Matrix Factorization (SGD)\n'
        '- 5-fold cross-validated\n'
        '- GridSearchCV tuned\n\n'
        '**Hybrid**\n'
        '- Wide candidate pool (top-50 by content)\n'
        '- Re-ranked by hybrid score\n'
        '- Tunable alpha slider'
    )

col1, col2 = st.columns([2, 1])

with col1:
    sorted_titles = sorted(movies['title'].tolist())
    default_title = 'Toy Story (1995)'
    default_idx   = sorted_titles.index(default_title) if default_title in sorted_titles else 0
    movie_title   = st.selectbox('Select a seed movie', options=sorted_titles, index=default_idx)

with col2:
    user_id = st.number_input(
        'User ID',
        min_value=1,
        max_value=int(ratings['userId'].max()),
        value=1,
        step=1,
        help=f'User IDs: 1 to {int(ratings["userId"].max())}'
    )

col_a, col_b = st.columns([3, 1])
with col_a:
    alpha = st.slider(
        'Content weight (alpha)',
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help='alpha=1.0 → pure content  |  alpha=0.0 → pure collaborative'
    )
with col_b:
    st.metric('Content', f'{alpha:.0%}')
    st.metric('Collab',  f'{1 - alpha:.0%}')

top_n = st.slider('Number of recommendations', min_value=5, max_value=20, value=10)

st.divider()

if st.button('Get Recommendations', type='primary', use_container_width=True):
    with st.spinner('Generating recommendations...'):
        results = hybrid_recommend(
            user_id     = user_id,
            movie_title = movie_title,
            top_n       = top_n,
            alpha       = alpha
        )

    if results is None:
        st.error(f"Movie '{movie_title}' not found in the dataset.")
    else:
        st.success(f'Top {top_n} recommendations for User {user_id} based on "{movie_title}"')

        st.dataframe(
            results,
            use_container_width=True,
            column_config={
                'Hybrid Score' : st.column_config.ProgressColumn(
                    'Hybrid Score',  min_value=0, max_value=1, format='%.3f'),
                'Content Score': st.column_config.ProgressColumn(
                    'Content Score', min_value=0, max_value=1, format='%.3f'),
                'Collab Score' : st.column_config.ProgressColumn(
                    'Collab Score',  min_value=0, max_value=1, format='%.3f'),
            }
        )

        st.caption(
            f'Hybrid Score = {alpha:.0%} × Content Score + {1-alpha:.0%} × Collab Score'
        )

        with st.expander('Score Breakdown Chart'):
            chart_df = results[['Title', 'Content Score', 'Collab Score', 'Hybrid Score']].head(10)
            chart_df = chart_df.set_index('Title')
            st.bar_chart(chart_df)

st.divider()
st.caption(
    'Dataset: MovieLens 100K  ·  '
    'Content: TF-IDF + Cosine Similarity  ·  '
    'Collaborative: Surprise SVD  ·  '
    'Hybrid: Weighted Average with wide candidate pool'
)
