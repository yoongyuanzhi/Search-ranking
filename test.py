from flashrank.Ranker import Ranker, RerankRequest

import pandas as pd

# ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/C:/Users/YZ/OneDrive - Singapore Management University/flashrank")


# Instantiate Ranker with absolute path
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/opt")

# Metadata is optional, Id can be your DB ids from your retrieval stage or simple numeric indices.
query = "retail shops approved"
passages = [
    {
        "id": 1,
        "text": "Despite a promising proposal, Case 1's retail shop development failed to meet certain urban development requirements. While the proposed location exhibited potential for economic revitalization, concerns were raised regarding its compatibility with existing zoning regulations. Additionally, the design lacked integration with surrounding architectural styles, detracting from the area's aesthetic cohesion. Although efforts were made to address accessibility issues, including pedestrian pathways and parking facilities, they fell short of the municipality's standards. Consequently, the development application for Case 1's retail shop was not approved. However, the applicant has been encouraged to revise the proposal, addressing the identified shortcomings for future consideration."
    },
    
    {
        "id": 2,
        "text": "In Case 2, the proposed retail shop underwent a rigorous evaluation process and was ultimately approved for development. The site selection demonstrated foresight, strategically positioning the shop within a thriving commercial hub. Architectural designs complemented the surrounding urban landscape, enhancing the area's visual appeal. Accessibility considerations, such as ample parking and provisions for disabled individuals, were meticulously integrated into the development plans. Moreover, the proposed retail activities aligned seamlessly with zoning regulations, fostering economic growth and community engagement. Case 2 exemplifies a successful collaboration between developers and urban planners, resulting in a retail space poised to enrich the local neighborhood."
    },
    {
        "id": 3,
        "text": "Despite initial concerns, Case 3's retail shop development was granted approval under specified conditions and exceptions. While the proposed location posed challenges in terms of traffic management and pedestrian flow, the applicant demonstrated a commitment to mitigating these issues through innovative design solutions. Flexible zoning regulations allowed for deviations from standard requirements, provided certain criteria were met. The retail concept showcased adaptability to evolving consumer trends, contributing to the area's commercial vibrancy. Through negotiation and compromise, Case 3's development proposal successfully navigated regulatory hurdles, paving the way for a unique retail experience that balances community needs with urban development objectives."
    },
        {
        "id": 4,
        "text": "Case 4 revolves around the establishment of a new park in the suburban area. Despite initial opposition from some local residents, the project gained approval due to its potential to enhance community well-being. The park's design incorporates sustainable features such as solar-powered lighting and rainwater harvesting systems, aligning with the municipality's green initiatives. Collaborative efforts between landscape architects and environmental experts ensured the preservation of natural habitats within the park grounds, providing a sanctuary for native flora and fauna."
    },
    {
        "id": 5,
        "text": "In Case 5, a tech startup sought permission to convert an old warehouse into a state-of-the-art innovation hub. The proposal garnered support from city officials eager to attract tech talent and foster entrepreneurship. Renovation plans include modern amenities such as co-working spaces, high-speed internet, and recreational facilities to create an inspiring work environment. The project's success hinges on its ability to revitalize an underutilized urban space while catalyzing economic growth through technology-driven initiatives."
    },
    {
        "id": 6,
        "text": "retail retail retail not approved not approved"
    },
        {
        "id": 7,
        "text": "Case 7 involves the construction of a new sports stadium on the outskirts of town. Despite initial concerns about noise pollution and traffic congestion, the project received widespread support from sports enthusiasts and local businesses eager to capitalize on game-day revenue. State-of-the-art facilities, including VIP boxes and high-definition video screens, are set to elevate the spectator experience, attracting major sporting events and boosting tourism in the region."
    },
    {
        "id": 8,
        "text": "In Case 8, a community garden initiative aimed to transform vacant lots into vibrant green spaces. Through grassroots organizing and volunteer efforts, neglected areas were revitalized with vegetable patches, flower beds, and communal gathering spaces. The project not only promotes sustainable living and food security but also fosters social cohesion by bringing residents together to cultivate shared greenery and exchange gardening knowledge."
    },
    {
        "id": 9,
        "text": "Case 9 centers on the establishment of a pet-friendly cafe in the heart of the city. Pet owners rejoice as they can now enjoy their favorite brews and bites in the company of their furry companions. The cafe features a dedicated pet menu, complete with treats and refreshments tailored to canine and feline tastes. With cozy seating areas and a welcoming atmosphere, it quickly becomes a beloved hangout spot for both pets and their humans."
    },
    {
        "id": 10,
        "text": "In Case 10, an ambitious art installation transforms a disused warehouse into an immersive gallery space. Renowned artists from around the world converge to create larger-than-life installations that challenge perceptions and ignite imaginations. The exhibit blurs the lines between reality and fantasy, inviting visitors on a journey of self-discovery and artistic exploration. From interactive light displays to thought-provoking sculptures, the warehouse-turned-gallery becomes a beacon of creativity in the urban landscape."
    },
    {
        "id": 11,
        "text": "Case 11 involves the renovation of a historic lighthouse perched on a rugged coastline. Preservationists and maritime enthusiasts rally behind the project, recognizing the lighthouse's cultural heritage and navigational significance. The restoration effort aims to preserve the tower's iconic silhouette while upgrading its internal mechanisms with modern technology. Once complete, the lighthouse will once again guide ships safely through treacherous waters, serving as a beacon of hope for sailors and sightseers alike."
    },
    {
        "id": 12,
        "text": "In Case 12, an urban farm initiative transforms vacant city lots into productive agricultural spaces. Equipped with raised beds, drip irrigation systems, and composting facilities, these mini-farms produce a bounty of fresh fruits and vegetables year-round. Community members participate in planting, harvesting, and selling produce at local markets, fostering a sense of pride and self-sufficiency while promoting healthy eating habits and environmental stewardship."
    },
    {
        "id": 13,
        "text": "Case 13 involves the creation of a pop-up street art festival in the downtown district. Local and international artists converge to transform blank walls into vibrant canvases, showcasing a diverse range of artistic styles and cultural influences. From colorful murals to interactive installations, the festival breathes new life into neglected urban spaces, sparking conversations and inspiring creativity. Visitors are encouraged to explore the city streets and discover hidden gems around every corner."
    },
    {
        "id": 14,
        "text": "In Case 14, an innovative bike-sharing program aims to promote eco-friendly transportation and reduce traffic congestion. Docking stations are strategically placed throughout the city, allowing residents and tourists to rent bicycles for short trips or leisurely rides. With convenient payment options and user-friendly mobile apps, the program encourages people to embrace cycling as a fun and sustainable way to get around town, while also improving air quality and public health."
    },
    {
        "id": 15,
        "text": "Case 15 centers on the transformation of a neglected industrial site into a thriving cultural precinct. Abandoned warehouses and factories are repurposed into galleries, theaters, and creative studios, attracting artists, performers, and cultural entrepreneurs from near and far. The precinct becomes a hub of artistic innovation and community engagement, hosting exhibitions, performances, and workshops that celebrate the rich diversity of local culture and heritage."
    },
       {
        "id": 16,
        "text": "Case 16 focuses on the establishment of a technology incubator in a formerly industrial neighborhood. The initiative aims to foster innovation and entrepreneurship by providing startups with affordable office space, mentorship programs, and access to venture capital. By repurposing old warehouses into dynamic workspaces, the incubator attracts tech talent from across the region, spurring economic growth and revitalizing the local economy."
    },
    {
        "id": 17,
        "text": "In Case 17, a community-led beautification project transforms drab alleyways into vibrant pedestrian walkways. Local artists are commissioned to paint murals, install sculptures, and create interactive installations that reflect the neighborhood's cultural heritage and spirit. The revitalized alleys become popular gathering spots for residents and visitors alike, fostering a sense of pride and community identity."
    },
    {
        "id": 18,
        "text": "Case 18 involves the development of a waterfront promenade along the city's riverfront. The project aims to reclaim unused industrial land and create a scenic public space for leisure and recreation. With landscaped gardens, bike paths, and waterfront cafes, the promenade becomes a beloved destination for strolling, jogging, and enjoying panoramic views of the skyline."
    },
    {
        "id": 19,
        "text": "In Case 19, a derelict train station is repurposed into a bustling food market and cultural hub. Local vendors set up stalls selling fresh produce, gourmet street food, and artisanal crafts, attracting foodies and shoppers from near and far. Live music performances, art exhibitions, and cooking workshops add to the market's lively atmosphere, transforming the station into a vibrant community gathering place."
    },
    {
        "id": 20,
        "text": "Case 20 centers on the renovation of a historic library building into a modern learning center and co-working space. The renovated facility offers state-of-the-art amenities, including high-speed internet, multimedia labs, and flexible meeting rooms, catering to students, freelancers, and small businesses alike. With its blend of tradition and innovation, the library becomes a dynamic hub of knowledge exchange and collaboration."
    },
    {
        "id": 21,
        "text": "In Case 21, an urban reforestation initiative aims to increase green canopy coverage and mitigate the effects of climate change. Volunteers plant native trees and shrubs in parks, streets, and vacant lots, creating habitats for wildlife and improving air quality. Through community engagement and stewardship, the initiative inspires residents to take an active role in greening their neighborhoods and creating a more sustainable future."
    },
    {
        "id": 22,
        "text": "Case 22 involves the redevelopment of a former industrial site into a mixed-use waterfront district. The project integrates residential, commercial, and recreational components, creating a vibrant live-work-play environment. With its pedestrian-friendly streets, waterfront promenades, and public parks, the district attracts residents, businesses, and tourists, catalyzing economic growth and revitalizing the surrounding area."
    },
    {
        "id": 23,
        "text": "In Case 23, a vacant lot is transformed into a community garden and outdoor classroom. Volunteers come together to plant organic vegetables, herbs, and flowers, while educators lead workshops on gardening, nutrition, and environmental stewardship. The garden becomes a place of learning and connection, fostering a sense of pride and ownership among participants while promoting healthy lifestyles and sustainability."
    },
    {
        "id": 24,
        "text": "Case 24 focuses on the adaptive reuse of a historic factory building into loft apartments and artist studios. The renovated spaces retain industrial charm with exposed brick walls, timber beams, and large windows, attracting creatives and professionals seeking unique live-work environments. With its blend of heritage architecture and modern amenities, the development adds vibrancy to the neighborhood while preserving its industrial heritage."
    },
    {
        "id": 25,
        "text": "In Case 25, an abandoned railway corridor is repurposed into a scenic greenway and multi-use trail. The transformed corridor provides a safe and accessible route for pedestrians and cyclists to explore the city's diverse neighborhoods and natural landscapes. With landscaped gardens, public art installations, and rest areas, the greenway enhances quality of life and encourages active transportation."
    },
    {
        "id": 26,
        "text": "Case 26 involves the construction of a mixed-income housing development in a gentrifying neighborhood. The project includes affordable housing units alongside market-rate apartments, promoting socioeconomic diversity and inclusive urban growth. With its emphasis on community amenities and sustainable design, the development sets a new standard for equitable housing development in the city."
    },
    {
        "id": 27,
        "text": "In Case 27, a neglected riverfront is revitalized into a dynamic cultural and recreational destination. The waterfront park features outdoor amphitheaters, art installations, and waterfront cafes, attracting visitors for concerts, festivals, and leisure activities. The revitalized riverfront becomes a source of civic pride and economic opportunity, driving tourism and investment in the surrounding area."
    },
    {
        "id": 28,
        "text": "Case 28 centers on the adaptive reuse of a decommissioned prison complex into a mixed-use development. The repurposed facility includes residential lofts, office space, and retail outlets, breathing new life into the historic site while preserving its architectural heritage. With its blend of old and new, the development becomes a catalyst for urban renewal and community revitalization."
    },
    {
        "id": 29,
        "text": "In Case 29, a neglected brownfield site is remediated and transformed into an eco-industrial park. The park attracts green businesses and clean technology startups, leveraging sustainable practices to minimize environmental impact and maximize resource efficiency. With its emphasis on innovation and sustainability, the eco-industrial park becomes a model for green economic development in the region."
    },
    {
        "id": 30,
        "text": "Case 30 involves the creation of a pedestrian-friendly plaza in the city center. The car-free plaza features outdoor seating, public art, and interactive fountains, providing a vibrant gathering space for residents and visitors alike. With its lively atmosphere and cultural events, the plaza becomes a focal point for community engagement and urban life."
    },
    {
        "id": 31,
        "text": "In Case 31, a neglected waterfront is revitalized into a thriving marine park and recreational harbor. The park features marinas, boat launches, and waterfront promenades, attracting boaters, anglers, and nature enthusiasts. With its emphasis on conservation and recreation, the marine park becomes a popular destination for outdoor enthusiasts and waterfront activities."
    },
    {
        "id": 32,
        "text": "Case 32 centers on the adaptive reuse of a historic train depot into a transportation museum and cultural center. The renovated depot showcases exhibits on the region's transportation history, while also serving as a venue for community events and educational programs. With its blend of history and culture, the museum becomes a destination for tourists and locals alike."
    },
    {
        "id": 33,
        "text": "In Case 33, a vacant lot is transformed into a vibrant community park and urban farm. The park features playgrounds, walking trails, and picnic areas, while the farm produces fresh fruits, vegetables, and herbs for local residents. With its emphasis on sustainability and community engagement, the park becomes a cherished green space in the neighborhood."
    },
    {
        "id": 34,
        "text": "Case 34 involves the renovation of a historic theater into a modern performing arts center. The renovated theater features state-of-the-art acoustics, lighting, and seating, attracting world-class performers and productions. With its blend of historic charm and modern amenities, the arts center becomes a cultural landmark and hub of artistic expression in the city."
    },
    {
        "id": 35,
        "text": "In Case 35, a neglected industrial site is repurposed into a sustainable business park. The park features green buildings, renewable energy systems, and eco-friendly landscaping, attracting environmentally conscious businesses and investors. With its emphasis on sustainability and innovation, the business park becomes a model for green economic development in the region."
    },
    {
        "id": 36,
        "text": "Case 36 centers on the development of a mixed-use transit-oriented development near a major transportation hub. The development includes residential units, office space, retail outlets, and public amenities, promoting walkability and reducing car dependency. With its convenient location and sustainable design, the development becomes a vibrant urban node and catalyst for economic growth."
    },
    {
        "id": 37,
        "text": "In Case 37, a neglected riverfront is transformed into a scenic greenbelt and recreational corridor. The greenbelt features walking trails, bike paths, and waterfront parks, providing opportunities for outdoor recreation and nature appreciation. With its emphasis on conservation and public access, the greenbelt becomes a cherished asset for the community and a haven for wildlife."
    },
    {
        "id": 38,
        "text": "Case 38 involves the adaptive reuse of a historic warehouse into a mixed-use development. The renovated warehouse features loft apartments, office space, and artisanal workshops, preserving its industrial character while accommodating modern uses. With its blend of history and innovation, the development becomes a vibrant hub of creativity and entrepreneurship in the city."
    },
    {
        "id": 39,
        "text": "In Case 39, a neglected brownfield site is remediated and transformed into a sustainable industrial park. The park attracts green industries and clean technology firms, leveraging renewable energy and resource-efficient practices to minimize environmental impact. With its emphasis on sustainability and innovation, the industrial park becomes a driver of green economic growth and job creation."
    },
    {
        "id": 40,
        "text": "Case 40 centers on the revitalization of a historic downtown district into a vibrant cultural and entertainment destination. The district features theaters, galleries, music venues, and restaurants, attracting visitors for live performances, art exhibitions, and culinary experiences. With its historic charm and lively atmosphere, the district becomes a magnet for tourists and a source of pride for locals."
    },
    {
        "id": 41,
        "text": "In Case 41, a neglected waterfront is transformed into a bustling commercial harbor and recreational marina. The harbor features docks, boat slips, and waterfront promenades, catering to boaters, fishermen, and waterfront enthusiasts. With its blend of commerce and recreation, the harbor becomes a vibrant hub of activity and a catalyst for waterfront revitalization."
    },
    {
        "id": 42,
        "text": "Case 42 involves the redevelopment of a former industrial site into a mixed-use waterfront district. The district features residential towers, office buildings, retail outlets, and public spaces, creating a dynamic live-work-play environment. With its waterfront location and sustainable design, the district becomes a sought-after destination for residents, businesses, and visitors alike."
    },
    {
        "id": 43,
        "text": "In Case 43, a neglected railway corridor is repurposed into a scenic rail-trail and linear park. The rail-trail features walking and biking paths, interpretive signage, and scenic overlooks, providing opportunities for outdoor recreation and nature appreciation. With its emphasis on conservation and public access, the rail-trail becomes a cherished asset for the community and a link between neighborhoods."
    },
    {
        "id": 44,
        "text": "Case 44 centers on the adaptive reuse of a historic school building into affordable housing and community facilities. The renovated school features apartments for low-income families, as well as spaces for childcare, adult education, and community events. With its blend of housing and services, the development becomes a vital resource for the neighborhood and a model for inclusive community development."
    },
    {
        "id": 45,
        "text": "In Case 45, a neglected brownfield site is remediated and transformed into a sustainable industrial park. The park attracts green industries and clean technology firms, leveraging renewable energy and resource-efficient practices to minimize environmental impact. With its emphasis on sustainability and innovation, the industrial park becomes a driver of green economic growth and job creation."
    },
    {
        "id": 46,
        "text": "Case 46 focuses on the development of a mixed-use transit-oriented development near a major transportation hub. The development includes residential units, office space, retail outlets, and public amenities, promoting walkability and reducing car dependency. With its convenient location and sustainable design, the development becomes a vibrant urban node and catalyst for economic growth."
    },
    {
        "id": 47,
        "text": "In Case 47, a neglected riverfront is transformed into a scenic greenbelt and recreational corridor. The greenbelt features walking trails, bike paths, and waterfront parks, providing opportunities for outdoor recreation and nature appreciation. With its emphasis on conservation and public access, the greenbelt becomes a cherished asset for the community and a haven for wildlife."
    },
    {
        "id": 48,
        "text": "Case 48 involves the adaptive reuse of a historic warehouse into a mixed-use development. The renovated warehouse features loft apartments, office space, and artisanal workshops, preserving its industrial character while accommodating modern uses. With its blend of history and innovation, the development becomes a vibrant hub of creativity and entrepreneurship in the city."
    },
    {
        "id": 49,
        "text": "In Case 49, a neglected brownfield site is remediated and transformed into a sustainable industrial park. The park attracts green industries and clean technology firms, leveraging renewable energy and resource-efficient practices to minimize environmental impact. With its emphasis on sustainability and innovation, the industrial park becomes a driver of green economic growth and job creation."
    },
    {
        "id": 50,
        "text": "Case 50 centers on the revitalization of a historic downtown district into a vibrant cultural and entertainment destination. The district features theaters, galleries, music venues, and restaurants, attracting visitors for live performances, art exhibitions, and culinary experiences. With its historic charm and lively atmosphere, the district becomes a magnet for tourists and a source of pride for locals."
    }

]
rerankrequest = RerankRequest(query=query, passages=passages)
results = ranker.rerank(rerankrequest)
# Convert the list of dictionaries to a pandas DataFrame
df = pd.DataFrame(results)

# Display the DataFrame
print(df)


