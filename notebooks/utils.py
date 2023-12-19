import umap
import altair as alt

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


def umap_plot(text, emb):

    cols = list(text.columns)
    # UMAP reduces the dimensions from 1024 to 2 dimensions that we can plot
    reducer = umap.UMAP(n_neighbors=2)
    umap_embeds = reducer.fit_transform(emb)
    # Prepare the data to plot and interactive visualization
    # using Altair
    # df_explore = pd.DataFrame(data={'text': qa['text']})
    # print(df_explore)

    # df_explore = pd.DataFrame(data={'text': qa_df[0]})
    df_explore = text.copy()
    df_explore['x'] = umap_embeds[:, 0]
    df_explore['y'] = umap_embeds[:, 1]

    # Plot
    chart = alt.Chart(df_explore).mark_circle(size=60).encode(
        x=  # 'x',
        alt.X('x',
              scale=alt.Scale(zero=False)
              ),
        y=alt.Y('y',
                scale=alt.Scale(zero=False)
                ),
        tooltip=cols
        # tooltip=['text']
    ).properties(
        width=700,
        height=400
    )
    return chart


def umap_plot_big(text, emb):

    cols = list(text.columns)
    # UMAP reduces the dimensions from 1024 to 2 dimensions that we can plot
    reducer = umap.UMAP(n_neighbors=100)
    umap_embeds = reducer.fit_transform(emb)
    # Prepare the data to plot and interactive visualization
    # using Altair
    # df_explore = pd.DataFrame(data={'text': qa['text']})
    # print(df_explore)

    # df_explore = pd.DataFrame(data={'text': qa_df[0]})
    df_explore = text.copy()
    df_explore['x'] = umap_embeds[:, 0]
    df_explore['y'] = umap_embeds[:, 1]

    # Plot
    chart = alt.Chart(df_explore).mark_circle(size=60).encode(
        x=  # 'x',
        alt.X('x',
              scale=alt.Scale(zero=False)
              ),
        y=alt.Y('y',
                scale=alt.Scale(zero=False)
                ),
        tooltip=cols
        # tooltip=['text']
    ).properties(
        width=700,
        height=400
    )
    return chart


def umap_plot_old(sentences, emb):
    # UMAP reduces the dimensions from 1024 to 2 dimensions that we can plot
    reducer = umap.UMAP(n_neighbors=2)
    umap_embeds = reducer.fit_transform(emb)
    # Prepare the data to plot and interactive visualization
    # using Altair
    # df_explore = pd.DataFrame(data={'text': qa['text']})
    # print(df_explore)

    # df_explore = pd.DataFrame(data={'text': qa_df[0]})
    df_explore = sentences
    df_explore['x'] = umap_embeds[:, 0]
    df_explore['y'] = umap_embeds[:, 1]

    # Plot
    chart = alt.Chart(df_explore).mark_circle(size=60).encode(
        x=  # 'x',
        alt.X('x',
              scale=alt.Scale(zero=False)
              ),
        y=alt.Y('y',
                scale=alt.Scale(zero=False)
                ),
        tooltip=['text']
    ).properties(
        width=700,
        height=400
    )
    return chart


def print_result(result):
    """ Print results with colorful formatting """
    for i, item in enumerate(result):
        print(f'item {i}')
        for key in item.keys():
            print(f"{key}:{item.get(key)}")
            print()
        print()

# Revised version


def keyword_search(query,
                   client,
                   results_lang='en',
                   properties=["title", "url", "text"],
                   num_results=3):

    where_filter = {
        "path": ["lang"],
        "operator": "Equal",
        "valueString": results_lang
    }

    response = (
        client.query.get("Articles", properties)
        .with_bm25(
            query=query
        )
        .with_where(where_filter)
        .with_limit(num_results)
        .do()
    )

    result = response['data']['Get']['Articles']
    return result


def dense_retrieval(query,
                    client,
                    results_lang='en',
                    properties=["text", "title", "url", "views",
                                "lang", "_additional {distance}"],
                    num_results=5):

    nearText = {"concepts": [query]}

    # To filter by language
    where_filter = {
        "path": ["lang"],
        "operator": "Equal",
        "valueString": results_lang
    }
    response = (
        client.query
        .get("Articles", properties)
        .with_near_text(nearText)
        .with_where(where_filter)
        .with_limit(num_results)
        .do()
    )

    result = response['data']['Get']['Articles']

    return result
