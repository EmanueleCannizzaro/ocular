# Template Integration Guide

## Overview
This guide documents the comprehensive cross-template integration system implemented in Ocular OCR Service. The system provides seamless navigation, consistent UI/UX, workflow persistence, and contextual actions across all pages.

## Architecture

### 1. Navigation Macros (`templates/components/navigation_macros.html`)

Reusable Jinja2 macros for consistent UI components across all templates.

#### Available Macros:

**`render_breadcrumbs(items)`**
```jinja2
{{ nav.render_breadcrumbs([
    {'label': 'Deployments', 'url': '/deployments', 'icon': 'fas fa-rocket'},
    {'label': 'GPU Models', 'url': '/models', 'icon': 'fas fa-cube'},
    {'label': 'Model Configuration', 'icon': 'fas fa-cog'}
]) }}
```

**`quick_action_card(title, icon, description, url, badge, color)`**
```jinja2
{{ nav.quick_action_card(
    title='Property Extraction',
    icon='fas fa-home',
    description='Extract structured data from property documents',
    url='/api/property-extraction/schema',
    badge={'text': 'NEW', 'color': 'danger'},
    color='danger'
) }}
```

**`contextual_help(section, tips)`**
```jinja2
{{ nav.contextual_help(
    section='Document Upload',
    tips=[
        'Drag and drop multiple files',
        'Use fallback strategy for reliability',
        'Add custom prompts for extraction'
    ]
) }}
```

**`workflow_progress(steps, current_step)`**
```jinja2
{{ nav.workflow_progress(
    steps=[
        {'icon': 'fas fa-upload', 'label': 'Upload PDF'},
        {'icon': 'fas fa-eye', 'label': 'OCR Processing'},
        {'icon': 'fas fa-brain', 'label': 'AI Extraction'},
        {'icon': 'fas fa-download', 'label': 'Download JSON'}
    ],
    current_step=1
) }}
```

**`related_features(features)`**
```jinja2
{{ nav.related_features([
    {
        'title': 'Providers Management',
        'description': 'Configure OCR providers',
        'url': '/providers',
        'icon': 'fas fa-cloud',
        'color': 'primary',
        'badge': {'text': 'Active', 'color': 'success'}
    }
]) }}
```

**`stats_widget(stats)`**
```jinja2
{{ nav.stats_widget([
    {
        'icon': 'fas fa-file-alt',
        'value': '1,234',
        'label': 'Documents Processed',
        'color': 'primary',
        'change': 12
    }
]) }}
```

**Other Macros:**
- `feature_status_badge(status)` - Display status badges (active, pending, error)
- `empty_state(title, description, icon, action_url, action_text)` - Empty state placeholders
- `page_header(title, subtitle, icon, actions)` - Page headers with action buttons
- `tab_navigation(tabs, active_tab_id)` - Tab navigation with state persistence
- `loading_skeleton(type, count)` - Loading skeletons (card, table, text)

### 2. Cross-Integration JavaScript (`static/js/cross_integration.js`)

#### CrossIntegrationManager Class

**Features:**
- Navigation tracking and history
- Workflow state persistence
- Quick actions floating button
- Contextual help panel
- Keyboard shortcuts
- Recent documents tracking
- Tab state management
- Auto-save (every 30 seconds)

**Keyboard Shortcuts:**
- `Alt+Q` - Open quick actions modal
- `Alt+H` - Toggle contextual help
- `Alt+N` - Show navigation history

**Usage:**
```javascript
// Add recent document
window.addRecentDocument({
    name: 'document.pdf',
    type: 'pdf',
    url: '/document-history/123'
});

// Save tab state
window.saveTabState('unified');

// Get recent documents
const recentDocs = window.crossIntegration.getRecentDocuments();
```

**Workflow State Management:**
The manager automatically saves:
- Form data (excluding passwords and files)
- Scroll position
- Active tab
- Selected provider
- Uploaded files info

State is restored when:
- Returning to same page within 1 hour
- Browser reload
- Navigation back

### 3. Styling (`static/css/cross_integration.css`)

**Key Features:**
- Smooth animations and transitions
- Hover effects on interactive elements
- Loading skeleton animations
- Workflow progress indicators
- Responsive design
- Dark mode support (prefers-color-scheme)
- Gradient backgrounds
- Icon animations

**Utility Classes:**
```css
.hover-lift          /* Card hover effect */
.cursor-pointer      /* Pointer cursor */
.transition-all      /* Smooth transitions */
.bg-gradient-primary /* Gradient background */
.text-shadow         /* Text shadow */
.box-shadow-lg       /* Large box shadow */
.fade-in-up          /* Fade in animation */
.slide-in-right      /* Slide in animation */
```

## Implementation Guide

### Step 1: Update Template to Use Macros

```jinja2
{% extends "base.html" %}
{% import "components/navigation_macros.html" as nav %}

{% block content %}
<!-- Page Header -->
{{ nav.page_header(
    title='Your Page Title',
    subtitle='Page description',
    icon='fas fa-icon',
    actions=[...]
) }}

<!-- Breadcrumbs -->
{{ nav.render_breadcrumbs([...]) }}

<!-- Your content here -->

<!-- Contextual Help -->
{{ nav.contextual_help('Section Name', [...tips]) }}
{% endblock %}
```

### Step 2: Add Cross-Page Links

Use the quick action cards and related features panels to guide users to relevant pages:

```jinja2
<div class="row">
    {{ nav.quick_action_card(...) }}
    {{ nav.quick_action_card(...) }}
    {{ nav.quick_action_card(...) }}
</div>

<!-- Sidebar -->
<div class="col-lg-4">
    {{ nav.related_features([...]) }}
</div>
```

### Step 3: Track User Actions

```javascript
// Track document processing
document.getElementById('processBtn').addEventListener('click', () => {
    // ... processing logic

    // Track in cross-integration
    window.addRecentDocument({
        name: filename,
        type: 'pdf',
        url: `/results/${resultId}`
    });
});
```

### Step 4: Implement Workflow Progress

For multi-step workflows:

```jinja2
{{ nav.workflow_progress(
    steps=[
        {'icon': 'fas fa-step1', 'label': 'Step 1'},
        {'icon': 'fas fa-step2', 'label': 'Step 2'},
        {'icon': 'fas fa-step3', 'label': 'Step 3'}
    ],
    current_step=1  # 0-indexed
) }}
```

Update progress with JavaScript:
```javascript
function updateWorkflowStep(stepIndex) {
    // Re-render workflow progress
    // Or use DOM manipulation
}
```

## Examples

### Example 1: Enhanced Upload Page

See `templates/index_enhanced.html` for a complete example showing:
- Statistics overview
- Tab navigation with state persistence
- Quick actions sidebar
- Related features panel
- Contextual help
- Feature highlights grid

### Example 2: Multi-Step Form

```jinja2
{% extends "base.html" %}
{% import "components/navigation_macros.html" as nav %}

{% block content %}
{{ nav.page_header(
    title='Property Extraction Workflow',
    subtitle='Extract structured data from appraisal documents'
) }}

{{ nav.workflow_progress(
    steps=[
        {'icon': 'fas fa-upload', 'label': 'Upload'},
        {'icon': 'fas fa-cog', 'label': 'Configure'},
        {'icon': 'fas fa-play', 'label': 'Process'},
        {'icon': 'fas fa-check', 'label': 'Complete'}
    ],
    current_step=current_step
) }}

<form id="workflowForm">
    <!-- Form fields based on current_step -->
</form>

{{ nav.contextual_help('Step ' + (current_step + 1)|string, step_tips) }}
{% endblock %}
```

### Example 3: Dashboard with Statistics

```jinja2
{{ nav.stats_widget([
    {
        'icon': 'fas fa-file-alt',
        'value': total_documents,
        'label': 'Total Documents',
        'color': 'primary',
        'change': 12
    },
    {
        'icon': 'fas fa-check-circle',
        'value': success_rate + '%',
        'label': 'Success Rate',
        'color': 'success',
        'change': 2
    },
    {
        'icon': 'fas fa-clock',
        'value': avg_time + 's',
        'label': 'Avg Processing Time',
        'color': 'warning',
        'change': -5
    },
    {
        'icon': 'fas fa-dollar-sign',
        'value': '$' + total_cost,
        'label': 'Total Cost',
        'color': 'info'
    }
]) }}
```

## Best Practices

### 1. Consistent Navigation
- Always use breadcrumbs for deep pages
- Include related features panel on detail pages
- Use quick action cards on landing pages

### 2. Workflow Continuity
- Save workflow state automatically
- Show progress indicators for multi-step processes
- Provide "Resume workflow" options

### 3. User Guidance
- Add contextual help on complex pages
- Use tooltips for form fields
- Include empty state messages with actions

### 4. Performance
- Use loading skeletons during data fetching
- Lazy-load heavy components
- Cache recent documents in localStorage

### 5. Accessibility
- Use semantic HTML
- Include ARIA labels
- Support keyboard navigation
- Maintain color contrast ratios

## Keyboard Shortcuts Reference

| Shortcut | Action |
|----------|--------|
| `Alt+Q` | Quick actions modal |
| `Alt+H` | Toggle contextual help |
| `Alt+N` | Navigation history |
| `Tab` | Navigate through elements |
| `Enter` | Activate focused element |
| `Esc` | Close modals/panels |

## Customization

### Adding New Quick Actions

Edit `cross_integration.js`:

```javascript
getContextualQuickActions() {
    const actions = {
        '/your-page': [
            {
                icon: 'fas fa-icon',
                label: 'Action Label',
                url: '/target-page',
                color: 'primary'
            }
        ]
    };
    return actions[currentPage] || [];
}
```

### Adding New Contextual Tips

Edit `cross_integration.js`:

```javascript
getContextualTips() {
    const tips = {
        '/your-page': [
            'Tip 1',
            'Tip 2',
            'Tip 3'
        ]
    };
    return tips[currentPage] || [];
}
```

### Custom Macro

Create in `navigation_macros.html`:

```jinja2
{% macro your_custom_macro(param1, param2) %}
<div class="your-component">
    <!-- Your component HTML -->
</div>
{% endmacro %}
```

Use in templates:

```jinja2
{% import "components/navigation_macros.html" as nav %}
{{ nav.your_custom_macro('value1', 'value2') }}
```

## Testing

### Manual Testing Checklist

- [ ] Navigate between pages and verify breadcrumbs update
- [ ] Test quick actions button (Alt+Q)
- [ ] Test contextual help (Alt+H)
- [ ] Test navigation history (Alt+N)
- [ ] Fill form, reload page, verify state restored
- [ ] Switch tabs, navigate away, return, verify tab restored
- [ ] Upload files, verify recent documents tracked
- [ ] Test keyboard shortcuts
- [ ] Test on mobile/tablet (responsive)
- [ ] Test with screen reader (accessibility)

### Browser Testing

Test in:
- Chrome/Edge (Chromium)
- Firefox
- Safari
- Mobile browsers

## Migration Guide

### Updating Existing Templates

1. Add macro import:
```jinja2
{% import "components/navigation_macros.html" as nav %}
```

2. Replace custom components with macros:
```jinja2
<!-- Old -->
<div class="breadcrumb">...</div>

<!-- New -->
{{ nav.render_breadcrumbs([...]) }}
```

3. Add contextual help:
```jinja2
{{ nav.contextual_help('Page Section', [...tips]) }}
```

4. Add related features:
```jinja2
{{ nav.related_features([...]) }}
```

## Troubleshooting

### State Not Persisting
- Check browser localStorage is enabled
- Verify page URL matches workflow storage key
- Check console for JavaScript errors

### Macros Not Rendering
- Verify import statement is correct
- Check macro name spelling
- Ensure parameters are in correct format

### Quick Actions Not Showing
- Verify CrossIntegrationManager is initialized
- Check contextual actions are defined for current page
- Inspect console for errors

### Styles Not Applied
- Verify cross_integration.css is loaded
- Check for CSS conflicts
- Clear browser cache

## Future Enhancements

Planned improvements:
- [ ] Real-time collaboration indicators
- [ ] Advanced workflow analytics
- [ ] Customizable quick actions per user
- [ ] Voice commands integration
- [ ] Progressive web app features
- [ ] Offline mode support
- [ ] Multi-language support for UI strings
- [ ] Theme customization

## Support

For issues or questions:
- Check this guide first
- Review `index_enhanced.html` for examples
- Inspect browser console for errors
- Check GitHub issues: https://github.com/EmanueleCannizzaro/ocular/issues

---

**Version**: 1.0.0
**Last Updated**: 2025-10-09
**Author**: Claude Code
**License**: MIT
