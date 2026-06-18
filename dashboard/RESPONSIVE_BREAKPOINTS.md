# Responsive Breakpoint Reference Guide

## Tailwind Breakpoint Tiers

```
0px ──────── 320px ──────── 380px ───── 640px ──────── 1024px ───────── 1280px+
 │           │              │          │              │               │
 └─ DEFAULT  └─ xs:         └ sm:      └─ md:         └─ lg:          └─ xl:
   (mobile)     (small)       (tablet)   (desktop)      (large)       (extra)
```

## Component Sizing at Each Breakpoint

### Header / NavBar
| Breakpoint | Height | Logo | Text | Icon | Gap |
|-----------|--------|------|------|------|-----|
| 0-379px   | 44px   | 12px | xs   | 12px | 6px |
| 380-639px | 48px   | 14px | sm   | 14px | 8px |
| 640px+    | 48px   | 14px | sm   | 16px | 12px |

### Hero Title
| Breakpoint | Size | Example |
|-----------|------|---------|
| 0-319px   | 24px | "PageForge" (2-3 lines if long) |
| 320-379px | 24px | "PageForge" (1-2 lines) |
| 380-639px | 36px | "PageForge" (1 line) |
| 640-1023px| 48px | "PageForge" (1 line) |
| 1024px+   | 60px | "PageForge" (1 line) |

### Main Padding
| Breakpoint | Padding | Notes |
|-----------|---------|-------|
| 0-379px   | 12px    | 3.75% of 320px screen |
| 380-639px | 16px    | Balanced spacing |
| 640px+    | 24px    | Comfortable whitespace |

### Section/Card Padding
| Breakpoint | Padding | Grid Gap |
|-----------|---------|----------|
| 0-379px   | 12px    | 8px |
| 380-639px | 16px    | 12px |
| 640-1023px| 16px    | 12px |
| 1024px+   | 24px    | 16px |

### Grid Layouts

#### 2-Column Grid (e.g., Hero Stats, KPI Cards at 2-col)
```
0-639px:     640px+:
[A] [B]      [A] [B]
[C] [D]      [C] [D]
```
Gap: 8px → 12px at `xs:` → 16px at `sm:`

#### 3-Column Grid (System Design Cards)
```
0-1023px:    1024px+:
[A]          [A] [B] [C]
[B]          
[C]          
```
Responsive column count: `grid-cols-1 md:grid-cols-3`

#### 4-Column Grid (KPI at breakpoint sm+)
```
0-379px:     380-639px:       640px+:
[A] [B]      [A] [B]          [A] [B] [C] [D]
[C] [D]      [C] [D]
```
Responsive: `grid-cols-2 md:grid-cols-4`

### Text Sizes Progression

#### Labels / Caption Text
```
0-319px:  8px   (very small but readable)
320-379px: 9px  (readable on modern phones)
380-639px: 10px (comfortable)
640px+:   11px  (standard)
```

#### Body Text
```
0-319px:  10px  (tight but works)
320-379px: 11px (comfortable)
380-639px: 12px (good padding)
640px+:   14px  (relaxed)
```

#### Headings
```
0-319px:  18px  (text-lg equivalent)
320-379px: 20px (text-xl)
380-639px: 24px (text-2xl)
640-1023px: 30px (text-3xl)
1024px+:  40px  (text-4xl or larger)
```

### Chart Heights
| Breakpoint | Height | Reason |
|-----------|--------|--------|
| 0-639px   | 200-240px | Keep viewport friendly |
| 640-1023px| 240px     | Maintain readability |
| 1024px+   | 300px     | Full detail visibility |

### Touch Target Sizes
| Element | Min Size | Example |
|---------|----------|---------|
| Icon buttons | 32x32px | GitHub icon in header |
| Text links | 44x24px | "Source on GitHub" |
| Card buttons | 40x40px | Metric cards |
| Chart tooltips | Auto | Min 32px height |

---

## CSS Class Patterns Used

### Responsive Padding
```css
/* Default → xs: → sm: → md: */
p-3       /* 12px */
xs:p-4    /* 16px at 380px+ */
sm:p-6    /* 24px at 640px+ */
md:p-8    /* 32px at 1024px+ */

/* Used in sections, cards, containers */
```

### Responsive Gaps
```css
/* Grid and flex gaps */
gap-2     /* 8px */
xs:gap-3  /* 12px at 380px+ */
sm:gap-4  /* 16px at 640px+ */
```

### Responsive Text
```css
/* Font sizes */
text-[10px]      /* 10px base */
xs:text-[11px]   /* 11px at 380px+ */
sm:text-xs       /* 12px at 640px+ */

/* Or scales */
text-sm          /* 14px base */
xs:text-base     /* 16px at 380px+ */
sm:text-lg       /* 18px at 640px+ */
```

### Responsive Display
```css
/* Hide/show at breakpoints */
hidden          /* Hidden by default */
xs:inline       /* Show at 380px+ */
sm:inline       /* Show at 640px+ */
hidden sm:block /* Hide sm-, show sm+ */
```

---

## Device-to-Breakpoint Mapping

### Mobile Phones
```
Device              Viewport     Breakpoint
─────────────────────────────────────────────
iPhone SE / 8       320px        (default)
Samsung Galaxy A12  360px        (default)
iPhone 11 / XR      390px        xs:
iPhone X / 12       375px        xs:
iPhone 14 Pro Max   430px        xs:
Google Pixel 5a     412px        xs:
```

### Tablets
```
Device              Viewport     Breakpoint
─────────────────────────────────────────────
iPad Mini (2024)    768px        sm:
iPad (10th gen)     810px        md:
iPad Pro 11"        834px        md:
iPad Pro 12.9"      1024px       md: / lg:
```

### Desktops & Laptops
```
Device              Viewport     Breakpoint
─────────────────────────────────────────────
Laptop (13")        1280px       lg:
Laptop (15")        1440px       lg:
Desktop (24")       1920px       xl:
Desktop (27")       2560px       2xl:
```

---

## Common Responsive Patterns Used

### Pattern 1: Stack → 2-Col → 3-Col
```jsx
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
  {/* 0-639px: 1 column (stacked) */}
  {/* 640-1023px: 2 columns */}
  {/* 1024px+: 3 columns */}
</div>
```

### Pattern 2: Progressive Text Sizing
```jsx
<h1 className="text-3xl xs:text-4xl sm:text-5xl md:text-6xl">
  {/* 0-379px: 30px */}
  {/* 380-639px: 36px */}
  {/* 640-1023px: 48px */}
  {/* 1024px+: 60px */}
</h1>
```

### Pattern 3: Responsive Padding + Gap
```jsx
<section className="p-3 xs:p-4 sm:p-6">
  <div className="flex flex-col gap-2 xs:gap-3 sm:gap-4">
    {/* Padding and gaps both scale */}
  </div>
</section>
```

### Pattern 4: Hide/Show Progressive Content
```jsx
<div>
  <span className="text-zinc-600">Version</span>
  <span className="text-zinc-700 hidden xs:inline">·</span>
  <span className="text-zinc-600 hidden sm:inline">Full info...</span>
</div>
```

### Pattern 5: Icon + Text Stacking
```jsx
<div className="flex items-center gap-1 xs:gap-1.5 sm:gap-2">
  <Icon className="w-3 xs:w-4 sm:w-5" />
  <span className="text-[10px] xs:text-[11px] sm:text-xs">Label</span>
</div>
```

---

## Testing Checklist

When testing responsive layout, verify at each breakpoint:

### At 320px
- [ ] No horizontal scroll
- [ ] Text readable (min 10px for body)
- [ ] Cards/sections not cramped (min 12px padding)
- [ ] Grids stack to 1-2 columns max
- [ ] Touch targets min 32x32px

### At 380px
- [ ] `xs:` classes applied correctly
- [ ] Intermediate sizing visible
- [ ] Padding/gaps scaled to 16px
- [ ] Text more relaxed (11px+)
- [ ] 2-column grids fit well

### At 640px
- [ ] `sm:` classes applied correctly
- [ ] Multi-column layouts render
- [ ] Charts at full height
- [ ] Padding comfortable (16-24px)
- [ ] Desktop-like spacing

### At 1024px+
- [ ] `md:` and `lg:` classes active
- [ ] 3-column grids display
- [ ] Full information visible
- [ ] Charts detailed and readable
- [ ] Optimal spacing throughout

---

## Common Mistakes to Avoid

❌ **DON'T:** Use fixed widths/heights
```jsx
// Bad - breaks on mobile
<div className="w-400 h-300">
```

✅ **DO:** Use responsive widths with max-width
```jsx
// Good - responsive from 0 to max-width
<div className="w-full max-w-4xl">
```

---

❌ **DON'T:** Forget intermediate breakpoint
```jsx
// Bad - jumps from sm to md
<div className="p-2 md:p-6">
```

✅ **DO:** Add xs: tier between mobile and sm
```jsx
// Good - smooth progression
<div className="p-2 xs:p-3 sm:p-4 md:p-6">
```

---

❌ **DON'T:** Use single breakpoint for responsive text
```jsx
// Bad - too big jump
<h1 className="text-2xl sm:text-6xl">
```

✅ **DO:** Progressive scaling
```jsx
// Good - readable at every size
<h1 className="text-2xl xs:text-3xl sm:text-4xl md:text-5xl lg:text-6xl">
```

---

❌ **DON'T:** Hide content on mobile that's important
```jsx
// Bad - info lost on mobile
<span className="hidden sm:inline">Important info</span>
```

✅ **DO:** Reorganize for mobile, not hide
```jsx
// Good - show info in different way on mobile
<span className="text-[10px] sm:text-xs">Important info</span>
```

---

## Maintenance Notes

When adding new components:

1. **Always start with base styles** (320px mobile)
2. **Add xs: overrides** for 380px phones
3. **Add sm: overrides** for 640px tablets
4. **Add md: overrides** for 1024px desktops
5. **Test at all breakpoints** before committing

Reference the patterns above and existing components for consistency.
