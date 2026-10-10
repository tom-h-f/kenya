// Generated from schema.json by json-schema-to-typescript. Do not edit; edit schema.json and regenerate (see README.md).

export interface Stamp {
  schema_version: 1;
  run_id: string;
  window_id: string;
  window_start: string;
  window_end: string;
  window_days: number;
  /**
   * daily_runs.computed_at
   */
  computed_at: string;
  /**
   * when the exporter wrote this file
   */
  generated_at: string;
  params: Params;
}
/**
 * What the detector was asked, copied from daily_runs and coord2 constants. Rendered next to every count as 'surfaced under these parameters'.
 */
export interface Params {
  text_threshold: number;
  text_overlap: number;
  min_entities: number;
  /**
   * relevance floor on a community's mean kenya_model
   */
  min_kenya: number;
  relevance_model?: string;
  leiden_resolution: number;
  leiden_seed: number;
  min_community_size: number;
  listing_budget?: number;
  listing_per_group?: number;
  /**
   * per trace; null = not edge-filtered (fast_retweet)
   */
  edge_percentile: {
    [k: string]: number | null;
  };
}

/**
 * What the detector was asked, copied from daily_runs and coord2 constants. Rendered next to every count as 'surfaced under these parameters'.
 */
export type Trace = "co_retweet" | "co_url" | "hashtag_sequence" | "fast_retweet" | "text_similarity";

/**
 * bit 1 co_retweet, 2 co_url, 4 hashtag_sequence, 8 fast_retweet, 16 text_similarity
 */
export type TraceMask = number;

/**
 * X user id as string. Never a number: ids exceed 2^53.
 */
export type AccountId = string;

export type PostId = string;

export interface Manifest {
  kind: "manifest";
  schema_version: 1;
  latest_run: string;
  generated_at: string;
  method_url: string;
  corrections_url: string;
  /**
   * newest first, at most 30
   */
  runs: {
    run_id: string;
    window_id: string;
    window_end: string;
    computed_at: string;
  }[];
  /**
   * keyed by path relative to runs/{latest_run}/; the only listing the app may use
   */
  files: {
    [k: string]: {
      bytes: number;
      sha256: string;
      rows?: number;
    };
  };
  /**
   * sources that could not be read this run; each view named here renders 'not available'
   */
  unavailable?: {
    view: string;
    reason: string;
  }[];
}

export interface Overview {
  kind: "overview";
  stamp: Stamp;
  /**
   * id of the persistent method caveat text; the app owns the wording, the export only names its version
   */
  caveat_id: string;
  /**
   * Counts of what the detector surfaced UNDER stamp.params. Render as 'surfaced under these parameters', never 'N coordinated accounts'.
   */
  surfaced: {
    accounts_in_graph: number;
    edges_in_graph: number;
    communities: number;
    communities_kept: number;
    communities_unscored: number;
    accounts_in_kept_communities: number;
  };
  /**
   * edges per trace, from daily_runs.edges_*
   */
  trace_edges: {
    [k: string]: number;
  };
  /**
   * How many co-retweet edges rest on one shared retweet; the page states it beside any network view.
   */
  edge_basis?: {
    co_retweet_edges_in_kept: number;
    with_shared_ge_2: number;
    with_shared_ge_3: number;
  };
  /**
   * posts in the window, with denominators
   */
  window_corpus?: {
    post_rows: number;
    distinct_posts: number;
    distinct_authors: number;
    embedding_coverage: number | null;
    relevance_coverage: number | null;
  };
}
/**
 * The freshness stamp and the parameters, shown on every view. Never omitted.
 */
export interface CommunityIndex {
  kind: "community_index";
  stamp: Stamp;
  communities: CommunityRow[];
}
/**
 * The freshness stamp and the parameters, shown on every view. Never omitted.
 */
export interface CommunityRow {
  cid: number;
  /**
   * stable id: sha1 of the lineage root's sorted member ids, first 10 hex; carried across runs by membership overlap
   */
  lineage: string;
  /**
   * neutral, e.g. 'Community 07'. Never a descriptive name that characterises the members.
   */
  label: string;
  size: number;
  edges: number;
  eigenvalue: number;
  /**
   * share of the community's edges per trace, from daily_communities.share_*
   */
  trace_share: {
    [k: string]: number;
  };
  /**
   * mean share of members' scored posts the relevance model calls Kenyan
   */
  kenya_model?: number | null;
  relevance_coverage?: number | null;
  relevance_scored: boolean;
  /**
   * false = below the relevance floor. Dropped communities are listed with counts only; no members, graph or accounts are exported for them.
   */
  kept: boolean;
  listed?: number;
  /**
   * @maxItems 10
   */
  top_hashtags?:
    | []
    | [TagCount]
    | [TagCount, TagCount]
    | [TagCount, TagCount, TagCount]
    | [TagCount, TagCount, TagCount, TagCount]
    | [TagCount, TagCount, TagCount, TagCount, TagCount]
    | [TagCount, TagCount, TagCount, TagCount, TagCount, TagCount]
    | [TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount]
    | [TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount]
    | [TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount]
    | [TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount];
  /**
   * @maxItems 5
   */
  top_domains?:
    | []
    | [DomainCount]
    | [DomainCount, DomainCount]
    | [DomainCount, DomainCount, DomainCount]
    | [DomainCount, DomainCount, DomainCount, DomainCount]
    | [DomainCount, DomainCount, DomainCount, DomainCount, DomainCount];
}
export interface TagCount {
  tag: string;
  posts: number;
  accounts: number;
}
export interface DomainCount {
  domain: string;
  posts: number;
  accounts: number;
}

/**
 * One Leiden community of min_size+ accounts. `cid` is the per-run id and is NOT stable across runs; `lineage` is.
 */
export interface CommunityDetail {
  kind: "community_detail";
  stamp: Stamp;
  community: CommunityRow;
  /**
   * columns, sorted by within_score descending; every member of a kept community
   */
  members: {
    id: AccountId[];
    handle: string[];
    within_score: number[];
    centrality: number[];
    posts_in_window: number[];
    kenya_model: (number | null)[];
  };
  /**
   * members' posts per UTC created_at day in the window, split by kind. A count in a collected corpus, not a rate.
   */
  daily_posts: {
    dt: string;
    originals: number;
    replies: number;
    reposts: number;
  }[];
  /**
   * authors this community's members repost most, where the original is in the corpus; coverage stated
   *
   * @maxItems 20
   */
  amplified?:
    | []
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
          members: number;
        }
      ];
  /**
   * share of members' reposts whose original author is known
   */
  amplified_coverage?: number | null;
  graph_available: boolean;
}
/**
 * The freshness stamp and the parameters, shown on every view. Never omitted.
 */
export interface GraphMeta {
  kind: "graph_meta";
  stamp: Stamp;
  nodes: {
    cid: number;
    size: number;
    x: number;
    y: number;
  }[];
  edges: {
    a: number;
    b: number;
    edges: number;
    edges_shared_ge_2?: number;
  }[];
}
/**
 * The freshness stamp and the parameters, shown on every view. Never omitted.
 */
export interface Graph {
  kind: "graph";
  stamp: Stamp;
  scope: "all" | "community" | "community_thin";
  cid?: number;
  /**
   * plain-language rule that produced this edge set, rendered on the view
   */
  edge_rule: string;
  /**
   * algorithm and seed, e.g. igraph drl seed 0
   */
  layout?: string;
  nodes: {
    id: AccountId[];
    handle: string[];
    cid: number[];
    x: number[];
    y: number[];
    degree: number[];
  };
  edges: {
    s: number[];
    t: number[];
    /**
     * max similarity weight over the edge's traces; a cosine, not a probability
     */
    w: number[];
    mask: TraceMask[];
    /**
     * number of retweeted posts both accounts retweeted (co_retweet trace); 0 when the edge has no co_retweet trace
     */
    shared: number[];
  };
}
/**
 * The freshness stamp and the parameters, shown on every view. Never omitted.
 */
export interface Evidence {
  kind: "evidence";
  stamp: Stamp;
  cid: number;
  /**
   * key 'e{index}' or 't{index}' -> items, capped at 5 per edge, with the full count
   */
  edges: {
    [k: string]: {
      total: number;
      items: EvidenceItem[];
    };
  };
  /**
   * post id -> public post, for item.post references. Absent id = original not in the corpus; the app links out to the post instead.
   */
  posts: {
    [k: string]: PostRef;
  };
}
/**
 * The freshness stamp and the parameters, shown on every view. Never omitted.
 */
export interface EvidenceItem {
  trace: Trace;
  /**
   * co_retweet: the retweeted post both retweeted. fast_retweet: the original.
   */
  post?: string;
  /**
   * co_url: the shared URL as persisted
   */
  url?: string;
  /**
   * hashtag_sequence: the ordered, lower-cased tag sequence both posted
   */
  tags?: string[];
  post_a?: PostId;
  post_b?: PostId;
  /**
   * text_similarity: cosine of the two posts' embeddings; post_a and post_b are the pair
   */
  similarity?: number;
  /**
   * fast_retweet: seconds from original to the retweet, per account is in the posts map
   */
  gap_seconds?: number;
}
export interface PostRef {
  id: PostId;
  author_id: AccountId;
  author_handle: string;
  created_at: string;
  text: string;
  url: string;
  like_count?: number | null;
  repost_count?: number | null;
  reply_count?: number | null;
  /**
   * collected_at of the snapshot the counts come from
   */
  counts_as_of?: string | null;
}

export interface AccountShard {
  kind: "account_shard";
  stamp: Stamp;
  shard: number;
  accounts: {
    [k: string]: AccountCard;
  };
}
/**
 * The freshness stamp and the parameters, shown on every view. Never omitted.
 */
export interface AccountCard {
  id: AccountId;
  handle: string;
  cid: number;
  lineage?: string;
  within_score: number;
  centrality?: number;
  kenya_model?: number | null;
  posts_scored?: number;
  posts_in_window: number;
  reposts_in_window?: number;
  profile: {
    display_name: string | null;
    bio: string | null;
    followers: number | null;
    following: number | null;
    statuses: number | null;
    created_at: string | null;
    /**
     * collected_at of the profile snapshot
     */
    profile_as_of: string | null;
  };
  /**
   * strongest linked accounts under the stamp's edge rule (shared>=2 or non-co_retweet), at most 25, with the evidence count so the card links to the evidence
   *
   * @maxItems 25
   */
  neighbours: {
    id: AccountId;
    handle: string;
    cid: number;
    mask: TraceMask;
    shared: number;
    w: number;
  }[];
  /**
   * all fused edges of this account, including single-shared-retweet ones
   */
  neighbours_total?: number;
  /**
   * @maxItems 10
   */
  hashtags?:
    | []
    | [TagCount]
    | [TagCount, TagCount]
    | [TagCount, TagCount, TagCount]
    | [TagCount, TagCount, TagCount, TagCount]
    | [TagCount, TagCount, TagCount, TagCount, TagCount]
    | [TagCount, TagCount, TagCount, TagCount, TagCount, TagCount]
    | [TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount]
    | [TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount]
    | [TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount]
    | [TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount];
  /**
   * authors this account reposts most, where the original is in the corpus
   *
   * @maxItems 10
   */
  amplifies?:
    | []
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        }
      ]
    | [
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        },
        {
          id: AccountId;
          handle: string;
          reposts: number;
        }
      ];
  daily_posts?: {
    dt: string;
    n: number;
  }[];
  /**
   * latest 10 non-repost posts created in the window
   *
   * @maxItems 10
   */
  recent_posts:
    | []
    | [PostRef]
    | [PostRef, PostRef]
    | [PostRef, PostRef, PostRef]
    | [PostRef, PostRef, PostRef, PostRef]
    | [PostRef, PostRef, PostRef, PostRef, PostRef]
    | [PostRef, PostRef, PostRef, PostRef, PostRef, PostRef]
    | [PostRef, PostRef, PostRef, PostRef, PostRef, PostRef, PostRef]
    | [PostRef, PostRef, PostRef, PostRef, PostRef, PostRef, PostRef, PostRef]
    | [PostRef, PostRef, PostRef, PostRef, PostRef, PostRef, PostRef, PostRef, PostRef]
    | [PostRef, PostRef, PostRef, PostRef, PostRef, PostRef, PostRef, PostRef, PostRef, PostRef];
}
export interface SearchAccounts {
  kind: "search_accounts";
  stamp: Stamp;
  id: AccountId[];
  handle: string[];
  display_name?: (string | null)[];
  cid: number[];
}
/**
 * The freshness stamp and the parameters, shown on every view. Never omitted.
 */
export interface HashtagIndex {
  kind: "hashtag_index";
  stamp: Stamp;
  basis: string;
  tags: {
    tag: string;
    posts: number;
    accounts: number;
    shard: number;
    first_seen?: string;
    emerging?: boolean;
  }[];
  /**
   * from trend_candidates, latest run: tags whose recent rate exceeds the prior rate in the control arm
   */
  emerging?: {
    tag: string;
    recent_rate: number;
    prior_rate: number;
    emergence: number;
    posts: number;
    censused?: boolean;
  }[];
  emerging_run?: string | null;
}
/**
 * The freshness stamp and the parameters, shown on every view. Never omitted.
 */
export interface HashtagShard {
  kind: "hashtag_shard";
  stamp: Stamp;
  shard: number;
  tags: {
    [k: string]: {
      tag: string;
      daily: {
        dt: string;
        posts: number;
        accounts: number;
      }[];
      /**
       * distribution of the tag's posts by author's community in the corpus window; includes 'none' for authors outside kept communities, so it is a share of tag posts and never a share of a community
       */
      communities: {
        cid: number | null;
        posts: number;
      }[];
      /**
       * @maxItems 5
       */
      examples:
        | []
        | [PostRef]
        | [PostRef, PostRef]
        | [PostRef, PostRef, PostRef]
        | [PostRef, PostRef, PostRef, PostRef]
        | [PostRef, PostRef, PostRef, PostRef, PostRef];
    };
  };
}
/**
 * The freshness stamp and the parameters, shown on every view. Never omitted.
 */
export interface LinkIndex {
  kind: "link_index";
  stamp: Stamp;
  /**
   * @maxItems 300
   */
  domains: DomainCount[];
  /**
   * @maxItems 500
   */
  urls: {
    url: string;
    posts: number;
    accounts: number;
    top_cids?: number[];
  }[];
}
/**
 * The freshness stamp and the parameters, shown on every view. Never omitted.
 */
export interface TopicIndex {
  kind: "topic_index";
  stamp: Stamp;
  fit: {
    fitted_at: string;
    fit_posts: number;
    topics: number;
    noise_share: number;
    params: string;
    /**
     * how posts after fitted_at are placed, with its similarity cut
     */
    assignment?: string;
  };
  topics: {
    tid: number;
    terms: string[];
    posts_in_window: number;
    accounts_in_window?: number;
    /**
     * posts per day over the window, in window order
     */
    daily: number[];
  }[];
}
/**
 * The freshness stamp and the parameters, shown on every view. Never omitted.
 */
export interface TopicDetail {
  kind: "topic_detail";
  stamp: Stamp;
  tid: number;
  terms: string[];
  /**
   * @maxItems 10
   */
  examples:
    | []
    | [PostRef]
    | [PostRef, PostRef]
    | [PostRef, PostRef, PostRef]
    | [PostRef, PostRef, PostRef, PostRef]
    | [PostRef, PostRef, PostRef, PostRef, PostRef]
    | [PostRef, PostRef, PostRef, PostRef, PostRef, PostRef]
    | [PostRef, PostRef, PostRef, PostRef, PostRef, PostRef, PostRef]
    | [PostRef, PostRef, PostRef, PostRef, PostRef, PostRef, PostRef, PostRef]
    | [PostRef, PostRef, PostRef, PostRef, PostRef, PostRef, PostRef, PostRef, PostRef]
    | [PostRef, PostRef, PostRef, PostRef, PostRef, PostRef, PostRef, PostRef, PostRef, PostRef];
  /**
   * topic mass by community: posts in this topic by members of each kept community, and that community's total posts for the denominator
   */
  communities: {
    cid: number;
    posts: number;
    community_posts: number;
  }[];
  /**
   * @maxItems 10
   */
  hashtags?:
    | []
    | [TagCount]
    | [TagCount, TagCount]
    | [TagCount, TagCount, TagCount]
    | [TagCount, TagCount, TagCount, TagCount]
    | [TagCount, TagCount, TagCount, TagCount, TagCount]
    | [TagCount, TagCount, TagCount, TagCount, TagCount, TagCount]
    | [TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount]
    | [TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount]
    | [TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount]
    | [TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount, TagCount];
  /**
   * AGGREGATE ONLY and only if Tom approves (open question 2): within-topic share of posts the hate classifier flags, with scored-post denominator and the scope it is computed within. Absent otherwise. Never per account, never per community.
   */
  flags?: {
    scope: "baseline" | "control";
    scored_posts: number;
    flagged_posts: number;
    model?: string;
    threshold_note?: string;
  };
}
/**
 * The freshness stamp and the parameters, shown on every view. Never omitted.
 */
export interface TimeCollection {
  kind: "time_collection";
  stamp: Stamp;
  series: {
    dt: string;
    rows: number;
    /**
     * rows per posts/ type= partition, so composition drift is visible
     */
    by_type: {
      [k: string]: number;
    };
  }[];
  gap_days: string[];
  control_arm?: {
    runs?: number;
    posts?: number;
    truncated_runs?: number;
    first?: string;
    last?: string;
  } | null;
}
/**
 * The freshness stamp and the parameters, shown on every view. Never omitted.
 */
export interface TimePipeline {
  kind: "time_pipeline";
  stamp: Stamp;
  /**
   * every persisted daily_runs row, newest first
   */
  runs: {
    run_id: string;
    window_id: string;
    computed_at: string;
    window_days: number;
    accounts_in_graph: number;
    edges_in_graph: number;
    communities: number;
    communities_kept: number;
    stage_seconds: {
      [k: string]: number;
    };
    edges?: {
      [k: string]: number;
    };
    params?: Params;
  }[];
}
/**
 * The freshness stamp and the parameters, shown on every view. Never omitted.
 */
export interface TimeCoverage {
  kind: "time_coverage";
  stamp: Stamp;
  denominator_posts: number;
  stages: {
    stage: "embeddings" | "relevance" | "hatespeech" | "incitement" | "labels";
    covered_posts: number;
    latest_dt?: string | null;
  }[];
}
/**
 * The freshness stamp and the parameters, shown on every view. Never omitted.
 */
export interface Changes {
  kind: "changes";
  stamp: Stamp;
  previous_run: string | null;
  min_jaccard?: number;
  matches: {
    status: "continuing" | "new" | "vanished" | "split" | "merged";
    lineage: string;
    cid?: number | null;
    previous_cid?: number | null;
    size?: number | null;
    previous_size?: number | null;
    jaccard?: number | null;
    joined?: number;
    left?: number;
  }[];
}
/**
 * The freshness stamp and the parameters, shown on every view. Never omitted.
 */
